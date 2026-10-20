#include "abyss-expert-stats.h"

#include "common.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr const char * k_topk_prefix = "ffn_moe_topk-";

struct phase_counts {
    std::vector<uint64_t>              tokens; // [layer]
    std::vector<std::vector<uint64_t>> hits;   // [layer][expert]
};

struct expert_stats {
    std::string  path;
    int          flush_sec = 30;
    std::mutex   mu;
    phase_counts phases[2]; // 0 = prefill, 1 = decode
    std::vector<uint8_t> buf;
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

void count(expert_stats & s, int il, const ggml_tensor * t) {
    const int64_t n_used   = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    // the top-k is usually a strided view of the argsort, so fetch the whole span once and stride on the host
    const size_t span = (size_t) (n_tokens - 1) * t->nb[1] + (size_t) n_used * t->nb[0];
    s.buf.resize(span);
    ggml_backend_tensor_get(t, s.buf.data(), 0, span);

    phase_counts & p = s.phases[n_tokens == 1 ? 1 : 0];
    if ((int) p.tokens.size() <= il) {
        p.tokens.resize(il + 1, 0);
        p.hits.resize(il + 1);
    }
    p.tokens[il] += n_tokens;

    std::vector<uint64_t> & hits = p.hits[il];
    for (int64_t j = 0; j < n_tokens; j++) {
        for (int64_t k = 0; k < n_used; k++) {
            int32_t e;
            memcpy(&e, s.buf.data() + j * t->nb[1] + k * t->nb[0], sizeof(e));
            if (e < 0) {
                continue;
            }
            if ((size_t) e >= hits.size()) {
                hits.resize(e + 1, 0);
            }
            hits[e]++;
        }
    }
}

bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    if (strncmp(t->name, k_topk_prefix, strlen(k_topk_prefix)) != 0 || t->type != GGML_TYPE_I32) {
        return !ask;
    }
    if (ask) {
        return true;
    }

    auto & s = *static_cast<expert_stats *>(user_data);
    const int il = atoi(t->name + strlen(k_topk_prefix));

    std::lock_guard<std::mutex> lock(s.mu);
    count(s, il, t);
    if (std::chrono::steady_clock::now() - s.last_flush >= std::chrono::seconds(s.flush_sec)) {
        flush_locked(s);
    }
    return true;
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
    if (params.cb_eval) {
        LOG_WRN("abyss: an eval callback is already set, expert stats disabled\n");
        return;
    }

    g_stats = new expert_stats();
    g_stats->path = path;
    if (const char * sec = getenv("ABYSS_EXPERT_STATS_FLUSH_SEC")) {
        g_stats->flush_sec = std::max(1, atoi(sec));
    }

    params.cb_eval           = eval_cb;
    params.cb_eval_user_data = g_stats;
    std::atexit(flush_at_exit);

    LOG_INF("abyss: counting routed experts into %s every %d s\n", g_stats->path.c_str(), g_stats->flush_sec);
}
