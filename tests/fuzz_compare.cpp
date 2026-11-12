// Differential fuzz test: OptimizedClassifier must always agree with LinearClassifier.
// Build:  make fuzz      Run:  ./fuzz_compare
#include "../baseline/linear_classifier.h"
#include "../starter/classifier.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace packet_rush;

static std::mt19937_64 rng;
static uint32_t rnd(uint32_t n) { return static_cast<uint32_t>(rng() % n); }
template <class T, size_t N> static T pick(const T (&a)[N]) { return a[rnd(N)]; }

// Small value pools so that many rules overlap and many packets match several rules.
static const uint32_t kIps[] = {0x0A000001, 0x0A000100, 0xC0A80001, 0x01020304, 0xFFFFFFFF, 0x00000000};
static const int kLens[] = {0, 8, 16, 24, 32};
static const uint16_t kPorts[] = {0, 1, 80, 443, 1024, 65535};
static const uint16_t kPktLen[] = {20, 40, 1500, 9000, 65535};
static const uint8_t kProto[] = {1, 6, 17, 0, 255};
static const uint8_t kFlagMask[] = {0x00, 0x02, 0x12, 0xFF, 0x80, 0x01};
static const uint8_t kSmall[] = {0, 1, 3, 8, 10, 63, 64, 255};
static const uint16_t kVlan[] = {0, 1, 10, 400, 4095};
static const uint16_t kRegion[] = {0, 1, 2, 16, 65535};
static const uint8_t kTtl[] = {0, 1, 30, 64, 128, 255};

static Rule make_rule(uint32_t id) {
    Rule r;
    r.rule_id = id;
    r.priority = rnd(40);                       // many ties on purpose
    r.action = static_cast<Action>(rnd(4));
    r.src_ip.set(pick(kIps), pick(kLens));
    r.dst_ip.set(pick(kIps), pick(kLens));
    auto range16 = [&](Range16& g) {
        if (rnd(100) < 40) { g.low = 0; g.high = 65535; return; }
        uint16_t a = pick(kPorts), b = pick(kPorts);
        g.low = std::min(a, b); g.high = std::max(a, b);
    };
    range16(r.src_port);
    range16(r.dst_port);
    if (rnd(100) < 40) { r.packet_len.low = 20; r.packet_len.high = 65535; }
    else { uint16_t a = pick(kPktLen), b = pick(kPktLen); r.packet_len.low = std::min(a, b); r.packet_len.high = std::max(a, b); }
    if (rnd(100) < 40) r.protocol.wildcard = true; else { r.protocol.wildcard = false; r.protocol.value = pick(kProto); }
    r.tcp_flags.mask = pick(kFlagMask);
    r.tcp_flags.value = static_cast<uint8_t>(rnd(256));
    if (rnd(100) < 40) r.region_id.wildcard = true; else { r.region_id.wildcard = false; r.region_id.value = pick(kRegion); }
    if (rnd(100) < 40) r.ingress_iface.wildcard = true; else { r.ingress_iface.wildcard = false; r.ingress_iface.value = pick(kSmall); }
    if (rnd(100) < 40) r.vlan_id.wildcard = true; else { r.vlan_id.wildcard = false; r.vlan_id.value = pick(kVlan); }
    if (rnd(100) < 40) r.dscp.wildcard = true; else { r.dscp.wildcard = false; r.dscp.value = pick(kSmall) & 63; }
    if (rnd(100) < 40) r.icmp_type.wildcard = true; else { r.icmp_type.wildcard = false; r.icmp_type.value = pick(kSmall); }
    if (rnd(100) < 40) { r.ttl.low = 0; r.ttl.high = 255; }
    else { uint8_t a = pick(kTtl), b = pick(kTtl); r.ttl.low = std::min(a, b); r.ttl.high = std::max(a, b); }
    return r;
}

static Packet make_packet() {
    Packet p{};
    p.src_ip = rnd(3) ? pick(kIps) + rnd(2) : static_cast<uint32_t>(rng());
    p.dst_ip = rnd(3) ? pick(kIps) + rnd(2) : static_cast<uint32_t>(rng());
    p.src_port = rnd(3) ? pick(kPorts) : static_cast<uint16_t>(rng());
    p.dst_port = rnd(3) ? pick(kPorts) : static_cast<uint16_t>(rng());
    p.packet_len = rnd(3) ? pick(kPktLen) : static_cast<uint16_t>(20 + rnd(65516));
    p.region_id = pick(kRegion);
    p.vlan_id = pick(kVlan);
    p.protocol = pick(kProto);
    p.tcp_flags = static_cast<uint8_t>(rnd(256));
    p.ingress_iface = pick(kSmall);
    p.dscp = pick(kSmall) & 63;
    p.icmp_type = pick(kSmall);
    p.ttl = rnd(3) ? pick(kTtl) : static_cast<uint8_t>(rng());
    return p;
}

static long g_checked = 0;
static bool compare(LinearClassifier& ref, OptimizedClassifier& opt, int npk, const char* where) {
    std::vector<Packet> pk(npk);
    for (auto& p : pk) p = make_packet();
    std::vector<Action> out(npk);
    opt.classify_batch(pk.data(), pk.size(), out.data());
    for (int i = 0; i < npk; ++i) {
        Action e = ref.classify(pk[i]);
        Action a = opt.classify(pk[i]);
        ++g_checked;
        if (a != e || out[i] != e) {
            std::printf("MISMATCH (%s): expected %d, got single=%d batch=%d\n", where, int(e), int(a), int(out[i]));
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int trials = argc > 1 ? std::atoi(argv[1]) : 60;
    const size_t sizes[] = {0, 1, 3, 63, 64, 65, 255, 256, 257, 1000, 3000};
    for (int t = 0; t < trials; ++t) {
        rng.seed(1234 + t);
        size_t n = sizes[t % (sizeof(sizes) / sizeof(sizes[0]))];
        std::vector<Rule> rules;
        uint32_t next_id = 1;
        for (size_t i = 0; i < n; ++i) rules.push_back(make_rule(next_id++));

        LinearClassifier ref;
        OptimizedClassifier opt;
        ref.initialize(rules);
        opt.initialize(rules);
        if (!compare(ref, opt, 300, "after initialize")) return 1;

        const int ops = 1200;
        for (int o = 0; o < ops; ++o) {
            uint32_t kind = rnd(10);
            if (kind < 4) {                       // insert a brand new rule
                Rule r = make_rule(next_id++);
                ref.insert_rule(r); opt.insert_rule(r);
            } else if (kind < 6) {                // overwrite an existing id
                Rule r = make_rule(1 + rnd(next_id));
                ref.insert_rule(r); opt.insert_rule(r);
            } else if (kind < 9) {                // delete (often an existing id, sometimes unknown)
                uint32_t id = rnd(5) ? 1 + rnd(next_id) : 1000000 + rnd(10);
                ref.delete_rule(id); opt.delete_rule(id);
            } else {                              // invalid insert: must be rejected, nothing changes
                Rule bad = make_rule(1 + rnd(next_id));
                bad.dst_port.low = 500; bad.dst_port.high = 100;
                opt.insert_rule(bad);             // reference NOT updated on purpose
            }
            if (o % 25 == 0 && !compare(ref, opt, 40, "after updates")) {
                std::printf("failed at trial %d op %d (n=%zu)\n", t, o, n);
                return 1;
            }
        }
        if (!compare(ref, opt, 500, "end of trial")) return 1;
    }
    std::printf("PASS: %d trials, %ld packet comparisons, no mismatches\n", trials, g_checked);
    return 0;
}
