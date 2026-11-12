#include "classifier.h"

#include <algorithm>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace packet_rush {

namespace {

constexpr uint32_t kDeltaFlag = 0x80000000u;
constexpr int kNumFields = 9;   // ids 0..7 are interval fields, id 8 is tcp_flags (masked)

inline bool rule_before(const Rule& a, const Rule& b) {
    if (a.priority != b.priority) return a.priority < b.priority;
    return a.rule_id < b.rule_id;
}

inline uint32_t field_domain(int id) { return id <= 2 ? 65536u : 256u; }

struct Interval {
    uint32_t lo = 0;
    uint32_t hi = 0;
    bool wild = false;   // matches every value
    bool empty = false;  // matches nothing (low > high)
};

// Which values of field `id` does this rule accept? (ids 0..7 only)
Interval rule_interval(const Rule& r, int id) {
    Interval iv;
    auto exact16 = [&](const ExactOrWildcard16& f) {
        if (f.wildcard) iv.wild = true; else iv.lo = iv.hi = f.value;
    };
    auto exact8 = [&](const ExactOrWildcard8& f) {
        if (f.wildcard) iv.wild = true; else iv.lo = iv.hi = f.value;
    };
    switch (id) {
        case 0: exact16(r.vlan_id); break;
        case 1: exact16(r.region_id); break;
        case 2:
            if (r.packet_len.low > r.packet_len.high) iv.empty = true;
            else { iv.lo = r.packet_len.low; iv.hi = r.packet_len.high; iv.wild = (iv.lo == 0 && iv.hi == 65535); }
            break;
        case 3: exact8(r.protocol); break;
        case 4: exact8(r.ingress_iface); break;
        case 5: exact8(r.dscp); break;
        case 6: exact8(r.icmp_type); break;
        case 7:
            if (r.ttl.low > r.ttl.high) iv.empty = true;
            else { iv.lo = r.ttl.low; iv.hi = r.ttl.high; iv.wild = (iv.lo == 0 && iv.hi == 255); }
            break;
        default: break;
    }
    return iv;
}

inline void extract_fields(const Packet& p, uint32_t* v) {
    v[0] = p.vlan_id;
    v[1] = p.region_id;
    v[2] = p.packet_len;
    v[3] = p.protocol;
    v[4] = p.ingress_iface;
    v[5] = p.dscp;
    v[6] = p.icmp_type;
    v[7] = p.ttl;
    v[8] = p.tcp_flags;
}

inline void set_bit(uint64_t* row, size_t i) { row[i >> 6] |= (uint64_t{1} << (i & 63)); }

} // namespace

OptimizedClassifier::OptimizedClassifier() = default;
OptimizedClassifier::~OptimizedClassifier() = default;

// ---------------------------------------------------------------------------
// Index construction
// ---------------------------------------------------------------------------
void OptimizedClassifier::build_index() {
    const size_t n = base_.size();
    nf_ = 0;
    for (auto& c : cls_) c.clear();

    stride_ = (((n + 63) / 64) + 3) & ~size_t(3);
    alive_.allocate(stride_);
    for (size_t i = 0; i < n; ++i) set_bit(alive_.p, i);
    if (n == 0) { arena_.allocate(0); return; }

    struct Plan {
        int id = 0;
        bool use = false;
        uint32_t rows = 1;                  // class 0 = "wildcard-only" row
        std::vector<uint32_t> cuts;         // interval fields: segment start values
        std::vector<uint16_t> segclass;     // interval fields: segment -> class
        std::vector<std::pair<uint8_t, uint8_t>> pats;  // flags: distinct (mask, value&mask)
        std::vector<uint64_t> classsig;     // flags: class -> bitmask of matching patterns
        std::vector<uint16_t> valclass;     // flags: value -> class
    };
    Plan plans[kNumFields];

    const size_t row_bytes = stride_ * sizeof(uint64_t);
    size_t budget_left = kArenaBudgetBytes;
    size_t total_rows = 0;

    // ---- planning: decide which fields to index and how many rows each needs ----
    for (int id = 0; id < kNumFields; ++id) {
        Plan& pl = plans[id];
        pl.id = id;
        if (id < 8) {
            const uint32_t dom = field_domain(id);
            size_t nonwild = 0, empties = 0;
            pl.cuts.assign(1, 0u);
            for (size_t i = 0; i < n; ++i) {
                Interval iv = rule_interval(base_[i], id);
                if (iv.empty) { ++empties; continue; }
                if (iv.wild) continue;
                ++nonwild;
                pl.cuts.push_back(iv.lo);
                if (iv.hi + 1 < dom) pl.cuts.push_back(iv.hi + 1);
            }
            if (nonwild == 0 && empties == 0) continue;
            std::sort(pl.cuts.begin(), pl.cuts.end());
            pl.cuts.erase(std::unique(pl.cuts.begin(), pl.cuts.end()), pl.cuts.end());
            const size_t nseg = pl.cuts.size();
            if (nseg > 8192) continue;

            std::vector<int32_t> diff(nseg + 1, 0);
            for (size_t i = 0; i < n; ++i) {
                Interval iv = rule_interval(base_[i], id);
                if (iv.empty || iv.wild) continue;
                size_t s_lo = std::lower_bound(pl.cuts.begin(), pl.cuts.end(), iv.lo) - pl.cuts.begin();
                size_t s_hi = (iv.hi + 1 < dom)
                                  ? size_t(std::lower_bound(pl.cuts.begin(), pl.cuts.end(), iv.hi + 1) - pl.cuts.begin())
                                  : nseg;
                ++diff[s_lo];
                --diff[s_hi];
            }
            pl.segclass.assign(nseg, 0);
            uint32_t next = 1;
            int32_t run = 0;
            for (size_t s = 0; s < nseg; ++s) {
                run += diff[s];
                if (run > 0) pl.segclass[s] = static_cast<uint16_t>(next++);
            }
            pl.rows = next;
        } else {
            // tcp_flags: (value & mask) == (rule.value & mask), mask == 0 is wildcard
            bool too_many = false;
            for (size_t i = 0; i < n && !too_many; ++i) {
                const Masked8& m = base_[i].tcp_flags;
                if (m.mask == 0) continue;
                std::pair<uint8_t, uint8_t> key{m.mask, static_cast<uint8_t>(m.value & m.mask)};
                if (std::find(pl.pats.begin(), pl.pats.end(), key) == pl.pats.end()) {
                    if (pl.pats.size() >= 64) too_many = true; else pl.pats.push_back(key);
                }
            }
            if (too_many || pl.pats.empty()) continue;
            pl.classsig.assign(1, 0ull);
            pl.valclass.assign(256, 0);
            for (uint32_t v = 0; v < 256; ++v) {
                uint64_t sig = 0;
                for (size_t k = 0; k < pl.pats.size(); ++k)
                    if ((v & pl.pats[k].first) == pl.pats[k].second) sig |= (uint64_t{1} << k);
                size_t c = std::find(pl.classsig.begin(), pl.classsig.end(), sig) - pl.classsig.begin();
                if (c == pl.classsig.size()) pl.classsig.push_back(sig);
                pl.valclass[v] = static_cast<uint16_t>(c);
            }
            pl.rows = static_cast<uint32_t>(pl.classsig.size());
        }
        const size_t bytes = size_t(pl.rows) * row_bytes;
        if (bytes > budget_left) continue;      // too big: skip this field (verification still exact)
        budget_left -= bytes;
        total_rows += pl.rows;
        pl.use = true;
    }

    arena_.allocate(total_rows * stride_);
    std::vector<uint64_t> wild(stride_);
    size_t row_cursor = 0;

    // ---- fill the bitsets ----
    for (int id = 0; id < kNumFields; ++id) {
        Plan& pl = plans[id];
        if (!pl.use) continue;
        uint64_t* base_row = arena_.p + row_cursor * stride_;
        std::fill(wild.begin(), wild.end(), 0ull);

        if (id < 8) {
            const uint32_t dom = field_domain(id);
            const size_t nseg = pl.cuts.size();
            for (size_t i = 0; i < n; ++i) {
                Interval iv = rule_interval(base_[i], id);
                if (iv.empty) continue;
                if (iv.wild) { set_bit(wild.data(), i); continue; }
                size_t s_lo = std::lower_bound(pl.cuts.begin(), pl.cuts.end(), iv.lo) - pl.cuts.begin();
                size_t s_hi = (iv.hi + 1 < dom)
                                  ? size_t(std::lower_bound(pl.cuts.begin(), pl.cuts.end(), iv.hi + 1) - pl.cuts.begin())
                                  : nseg;
                for (size_t s = s_lo; s < s_hi; ++s)
                    set_bit(base_row + size_t(pl.segclass[s]) * stride_, i);
            }
            std::vector<uint16_t> cls(dom, 0);
            for (size_t s = 0; s < nseg; ++s) {
                const uint32_t end = (s + 1 < nseg) ? pl.cuts[s + 1] : dom;
                for (uint32_t v = pl.cuts[s]; v < end; ++v) cls[v] = pl.segclass[s];
            }
            cls_[nf_] = std::move(cls);
        } else {
            for (size_t i = 0; i < n; ++i) {
                const Masked8& m = base_[i].tcp_flags;
                if (m.mask == 0) { set_bit(wild.data(), i); continue; }
                std::pair<uint8_t, uint8_t> key{m.mask, static_cast<uint8_t>(m.value & m.mask)};
                size_t pi = std::find(pl.pats.begin(), pl.pats.end(), key) - pl.pats.begin();
                for (uint32_t c = 0; c < pl.rows; ++c)
                    if ((pl.classsig[c] >> pi) & 1) set_bit(base_row + size_t(c) * stride_, i);
            }
            cls_[nf_] = pl.valclass;
        }
        // every row also accepts the rules that are wildcard on this field
        for (uint32_t r = 0; r < pl.rows; ++r) {
            uint64_t* row = base_row + size_t(r) * stride_;
            for (size_t w = 0; w < stride_; ++w) row[w] |= wild[w];
        }
        fid_[nf_] = id;
        row0_[nf_] = row_cursor;
        row_cursor += pl.rows;
        ++nf_;
    }
}

void OptimizedClassifier::rebuild() {
    // base_ is sorted already; sort the small overlay and merge the two.
    std::vector<Rule> merged;
    merged.reserve(base_.size() - dead_ + delta_.size());
    std::vector<Rule> add(delta_);
    std::sort(add.begin(), add.end(), rule_before);

    size_t ai = 0;
    for (size_t i = 0; i < base_.size(); ++i) {
        if (!((alive_.p[i >> 6] >> (i & 63)) & 1)) continue;
        while (ai < add.size() && rule_before(add[ai], base_[i])) merged.push_back(add[ai++]);
        merged.push_back(base_[i]);
    }
    while (ai < add.size()) merged.push_back(add[ai++]);

    base_.swap(merged);
    delta_.clear();
    dead_ = 0;
    where_.clear();
    where_.reserve(base_.size() * 2);
    for (size_t i = 0; i < base_.size(); ++i) where_[base_[i].rule_id] = static_cast<uint32_t>(i);
    build_index();
}

void OptimizedClassifier::initialize(const std::vector<Rule>& rules) {
    base_ = rules;
    std::sort(base_.begin(), base_.end(), rule_before);
    delta_.clear();
    dead_ = 0;
    where_.clear();
    where_.reserve(base_.size() * 2);
    for (size_t i = 0; i < base_.size(); ++i) where_[base_[i].rule_id] = static_cast<uint32_t>(i);
    build_index();
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------
const Rule* OptimizedClassifier::find_base(const Packet& p) const {
    if (base_.empty()) return nullptr;

    uint32_t v[kNumFields];
    extract_fields(p, v);

    const uint64_t* rows[kMaxFields];
    for (int k = 0; k < nf_; ++k)
        rows[k] = arena_.p + (row0_[k] + cls_[k][v[fid_[k]]]) * stride_;

    const uint64_t* alive = alive_.p;

#if defined(__AVX2__)
    for (size_t w = 0; w < stride_; w += 4) {
        __m256i acc = _mm256_load_si256(reinterpret_cast<const __m256i*>(alive + w));
        for (int k = 0; k < nf_; ++k)
            acc = _mm256_and_si256(acc, _mm256_load_si256(reinterpret_cast<const __m256i*>(rows[k] + w)));
        if (_mm256_testz_si256(acc, acc)) continue;
        alignas(32) uint64_t t[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(t), acc);
        for (int j = 0; j < 4; ++j) {
            uint64_t bits = t[j];
            while (bits) {
                const size_t idx = (w + j) * 64 + static_cast<size_t>(__builtin_ctzll(bits));
                if (base_[idx].matches(p)) return &base_[idx];
                bits &= bits - 1;
            }
        }
    }
#else
    for (size_t w = 0; w < stride_; ++w) {
        uint64_t bits = alive[w];
        for (int k = 0; k < nf_ && bits; ++k) bits &= rows[k][w];
        while (bits) {
            const size_t idx = w * 64 + static_cast<size_t>(__builtin_ctzll(bits));
            if (base_[idx].matches(p)) return &base_[idx];
            bits &= bits - 1;
        }
    }
#endif
    return nullptr;
}

Action OptimizedClassifier::classify(const Packet& packet) {
    const Rule* best = find_base(packet);
    for (const Rule& r : delta_) {
        if (r.matches(packet) && (best == nullptr || rule_before(r, *best))) best = &r;
    }
    return best ? best->action : Action::DROP;
}

void OptimizedClassifier::classify_batch(const Packet* packets, size_t count, Action* out_actions) {
    for (size_t i = 0; i < count; ++i) out_actions[i] = classify(packets[i]);
}

// ---------------------------------------------------------------------------
// Updates
// ---------------------------------------------------------------------------
void OptimizedClassifier::erase_id(uint32_t rule_id) {
    auto it = where_.find(rule_id);
    if (it == where_.end()) return;
    const uint32_t loc = it->second;
    if (loc & kDeltaFlag) {
        const size_t idx = loc & ~kDeltaFlag;
        const size_t last = delta_.size() - 1;
        if (idx != last) {
            delta_[idx] = delta_[last];
            where_[delta_[idx].rule_id] = kDeltaFlag | static_cast<uint32_t>(idx);
        }
        delta_.pop_back();
    } else {
        alive_.p[loc >> 6] &= ~(uint64_t{1} << (loc & 63));
        ++dead_;
    }
    where_.erase(rule_id);
}

void OptimizedClassifier::insert_rule(const Rule& rule) {
    // Invalid rules are rejected and leave the classifier untouched (spec section 6).
    if (!rule.is_valid()) return;
    erase_id(rule.rule_id);
    delta_.push_back(rule);
    where_[rule.rule_id] = kDeltaFlag | static_cast<uint32_t>(delta_.size() - 1);
    if (delta_.size() >= kDeltaLimit) rebuild();
}

void OptimizedClassifier::delete_rule(uint32_t rule_id) {
    erase_id(rule_id);
    if (dead_ > base_.size() / 2 + 1024) rebuild();
}

} // namespace packet_rush
