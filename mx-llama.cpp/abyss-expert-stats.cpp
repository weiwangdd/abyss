#include "abyss-expert-stats.h"

#include "common.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct phase_counts {
    std::vector<uint64_t>              tokens; // [layer]
    std::vector<std::vector<uint64_t>> hits;   // [layer][expert]
};

struct expert_stats {
    std::string  path;
    int          flush_sec = 30;
    std::mutex   mu;
    phase_counts phases[2]; // 0 = prefill, 1 = decode
    std::chrono::steady_clock::time_point last_flush = std::chrono::steady_clock::now();
};

expert_stats * g_stats = nullptr;

void write_phase(FILE * f, const phase_counts & p) {
    fprintf(f, "{\"tokens\":[");
    for (size_t il = 0; il < p.tokens.size(); il++) {
        fprintf(f, "%s%llu", il ? "," : "", (unsigned long long) p.tokens[il]);
    }
    fprintf(f, "],\"hits\":[");
    for (size_t il = 0; il < p.hits.size(); il++) {
        fprintf(f, "%s[", il ? "," : "");
        for (size_t e = 0; e < p.hits[il].size(); e++) {
            fprintf(f, "%s%llu", e ? "," : "", (unsigned long long) p.hits[il][e]);
        }
        fprintf(f, "]");
    }
    fprintf(f, "]}");
}

// caller holds mu; write to a temp file and rename so a reader never sees a partial file
void flush_locked(expert_stats & s) {
    const std::string tmp = s.path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "w");
    if (!f) {
        LOG_WRN("abyss: cannot write %s\n", tmp.c_str());
        return;
    }
    fprintf(f, "{\"version\":1,\"updated\":%lld,\"prefill\":", (long long) time(nullptr));
    write_phase(f, s.phases[0]);
    fprintf(f, ",\"decode\":");
    write_phase(f, s.phases[1]);
    fprintf(f, "}\n");
    fclose(f);
    rename(tmp.c_str(), s.path.c_str());
    s.last_flush = std::chrono::steady_clock::now();
}

void count(expert_stats & s, int il, const int32_t * ids, int64_t n_used, int64_t n_tokens) {
    phase_counts & p = s.phases[n_tokens == 1 ? 1 : 0];
    if ((int) p.tokens.size() <= il) {
        p.tokens.resize(il + 1, 0);
        p.hits.resize(il + 1);
    }
    p.tokens[il] += n_tokens;

    std::vector<uint64_t> & hits = p.hits[il];
    for (int64_t i = 0; i < n_tokens * n_used; i++) {
        const int32_t e = ids[i];
        if (e < 0) {
            continue;
        }
        if ((size_t) e >= hits.size()) {
            hits.resize(e + 1, 0);
        }
        hits[e]++;
    }
}

// the selections of a ubatch arrive at the synchronize after it, the graph is not cut for them like an eval callback
// on each layer's top-k would
void on_moe_ids(int32_t il, const int32_t * ids, int32_t n_used, int32_t n_tokens, void * user_data) {
    auto & s = *static_cast<expert_stats *>(user_data);

    std::lock_guard<std::mutex> lock(s.mu);
    count(s, il, ids, n_used, n_tokens);
    if (std::chrono::steady_clock::now() - s.last_flush >= std::chrono::seconds(s.flush_sec)) {
        flush_locked(s);
    }
}

// a restart keeps adding to the counts already on disk instead of overwriting them
void load_existing(expert_stats & s) {
    FILE * f = fopen(s.path.c_str(), "r");
    if (!f) {
        return;
    }
    nlohmann::json j = nlohmann::json::parse(f, nullptr, false);
    fclose(f);
    if (j.is_discarded() || j.value("version", 0) != 1) {
        LOG_WRN("abyss: %s is not a version 1 stats file, starting from zero\n", s.path.c_str());
        return;
    }
    const char * names[2] = { "prefill", "decode" };
    for (int ph = 0; ph < 2; ph++) {
        s.phases[ph].tokens = j[names[ph]]["tokens"].get<std::vector<uint64_t>>();
        s.phases[ph].hits   = j[names[ph]]["hits"].get<std::vector<std::vector<uint64_t>>>();
        s.phases[ph].hits.resize(s.phases[ph].tokens.size());
    }
    LOG_INF("abyss: continuing from %s\n", s.path.c_str());
}

void flush_at_exit() {
    if (g_stats) {
        std::lock_guard<std::mutex> lock(g_stats->mu);
        flush_locked(*g_stats);
    }
}

} // namespace

void abyss_expert_stats_install(common_params & params) {
    const char * path = getenv("ABYSS_EXPERT_STATS");
    if (!path || !*path) {
        return;
    }
    if (params.cb_moe_ids) {
        LOG_WRN("abyss: a MoE selection callback is already set, expert stats disabled\n");
        return;
    }

    g_stats = new expert_stats();
    g_stats->path = path;
    if (const char * sec = getenv("ABYSS_EXPERT_STATS_FLUSH_SEC")) {
        g_stats->flush_sec = std::max(1, atoi(sec));
    }
    load_existing(*g_stats);

    params.cb_moe_ids           = on_moe_ids;
    params.cb_moe_ids_user_data = g_stats;
    std::atexit(flush_at_exit);

    LOG_INF("abyss: counting routed experts into %s every %d s\n", g_stats->path.c_str(), g_stats->flush_sec);
}
