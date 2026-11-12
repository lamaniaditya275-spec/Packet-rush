#pragma once

#include "classifier_interface.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace packet_rush {

/**
 * Filter-and-verify packet classifier.
 *
 *  1. Rules are kept sorted by (priority, rule_id): rule index == precedence.
 *  2. For 9 small header fields we pre-compute bitsets (one bit per rule).
 *     A packet looks up one bitset per field, ANDs them with AVX2, and the set
 *     bits are the *candidate* rules, already in precedence order.
 *  3. Candidates are confirmed with the exact Rule::matches(); the first
 *     confirmed candidate is the winner. The filter may let extra candidates
 *     through but never drops a true match, so results equal the baseline.
 *  4. Updates go to a small overlay (delta list + "deleted" bits), and the
 *     index is rebuilt only occasionally.
 */
class OptimizedClassifier : public IPacketClassifier {
public:
    OptimizedClassifier();
    ~OptimizedClassifier() override;
    OptimizedClassifier(const OptimizedClassifier&) = delete;
    OptimizedClassifier& operator=(const OptimizedClassifier&) = delete;

    void initialize(const std::vector<Rule>& rules) override;
    Action classify(const Packet& packet) override;
    void classify_batch(const Packet* packets, size_t count, Action* out_actions) override;
    void insert_rule(const Rule& rule) override;
    void delete_rule(uint32_t rule_id) override;

    const char* name() const override { return "OptimizedClassifier"; }

private:
    static constexpr int kMaxFields = 9;
    static constexpr size_t kDeltaLimit = 256;             // rebuild after this many overlay rules
    static constexpr size_t kArenaBudgetBytes = 16u << 20; // max memory for filter bitsets

    // 32-byte aligned, zero-filled word buffer (needed for AVX2 loads).
    struct AlignedBuf {
        uint64_t* p = nullptr;
        AlignedBuf() = default;
        AlignedBuf(const AlignedBuf&) = delete;
        AlignedBuf& operator=(const AlignedBuf&) = delete;
        ~AlignedBuf() { std::free(p); }
        void allocate(size_t words) {
            std::free(p);
            p = nullptr;
            if (words == 0) return;
            size_t bytes = ((words * sizeof(uint64_t) + 31) / 32) * 32;
            p = static_cast<uint64_t*>(std::aligned_alloc(32, bytes));
            if (p) std::memset(p, 0, bytes);
        }
    };

    void build_index();
    void rebuild();
    void erase_id(uint32_t rule_id);
    const Rule* find_base(const Packet& packet) const;

    std::vector<Rule> base_;      // sorted by (priority, rule_id)
    std::vector<Rule> delta_;     // recently inserted rules (small, unsorted)
    std::unordered_map<uint32_t, uint32_t> where_;  // rule_id -> base index, or delta index | high bit
    size_t dead_ = 0;             // deleted-but-not-yet-compacted base rules

    size_t stride_ = 0;           // 64-bit words per bitset row (multiple of 4)
    AlignedBuf arena_;            // all filter bitset rows
    AlignedBuf alive_;            // bit i set => base_[i] is still active

    int nf_ = 0;                  // number of active filter fields
    int fid_[kMaxFields] = {};    // which packet field each filter uses
    size_t row0_[kMaxFields] = {};            // first arena row of each field
    std::vector<uint16_t> cls_[kMaxFields];   // field value -> bitset row (class)
};

} // namespace packet_rush
