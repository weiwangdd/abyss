// abyss-dump: run a prompt through the model and save the graph tensors whose names match a regex, for checking a
// layer against a reference implementation outside llama.cpp.
//
//   ABYSS_DUMP_RE='^idx_(q|k)-2$' ABYSS_DUMP_DIR=out llama-abyss-dump -m model.gguf -f prompt.txt -c 8192 -b 8192 ...
//
// Each evaluation of a matching tensor becomes <dir>/<name>.<n>.bin, n counting from 0 per name (one per ubatch):
// "ADMP", int32 ggml type, int64 ne[4], then the data as the tensor holds it (contiguous tensors only).
//
// With ABYSS_DUMP_GEN=N the prompt is not dumped; N tokens are then decoded greedily one at a time and only those
// single-token evaluations are dumped, for checking the decode path.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <regex>
#include <string>
#include <vector>

struct dump_state {
    std::regex re;
    std::string dir;
    std::map<std::string, int> seq;
    std::vector<uint8_t> buf;
    bool active = true;
};

static bool dump_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto & s = *static_cast<dump_state *>(user_data);
    if (!s.active || !std::regex_search(t->name, s.re)) {
        return !ask;
    }
    if (ask) {
        return true;
    }
    if (!ggml_is_contiguous(t)) {
        LOG_WRN("abyss-dump: %s is not contiguous, skipped\n", t->name);
        return true;
    }

    const size_t nbytes = ggml_nbytes(t);
    s.buf.resize(nbytes);
    ggml_backend_tensor_get(t, s.buf.data(), 0, nbytes);

    const int n = s.seq[t->name]++;
    const std::string path = s.dir + "/" + t->name + "." + std::to_string(n) + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LOG_ERR("abyss-dump: cannot write %s\n", path.c_str());
        return false;
    }
    const int32_t type = t->type;
    const int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    fwrite("ADMP", 1, 4, f);
    fwrite(&type, sizeof(type), 1, f);
    fwrite(ne, sizeof(ne), 1, f);
    fwrite(s.buf.data(), 1, nbytes, f);
    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * re  = getenv("ABYSS_DUMP_RE");
    const char * dir = getenv("ABYSS_DUMP_DIR");
    if (!re || !dir) {
        LOG_ERR("set ABYSS_DUMP_RE and ABYSS_DUMP_DIR\n");
        return 1;
    }
    dump_state state;
    state.re  = std::regex(re);
    state.dir = dir;

    const char * gen = getenv("ABYSS_DUMP_GEN");
    const int n_gen = gen ? atoi(gen) : 0;
    state.active = n_gen == 0;

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = dump_cb;
    params.cb_eval_user_data = &state;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init->context();
    if (llama_init->model() == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(llama_init->model());
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    LOG_INF("abyss-dump: %zu prompt tokens\n", tokens.size());

    // the whole prompt in one batch, so its ubatches follow each other as in a server prefill
    common_batch batch = common_batch_get_one(ctx, tokens);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("failed to eval\n");
        return 1;
    }

    // greedy decoding, one token per evaluation
    state.active = true;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < n_gen; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token tok = 0;
        for (llama_token v = 1; v < n_vocab; ++v) {
            if (logits[v] > logits[tok]) {
                tok = v;
            }
        }
        LOG_INF("abyss-dump: generated token %d: %d\n", i, tok);
        common_batch next = common_batch_get_one(ctx, { tok });
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, next.get())) {
            LOG_ERR("failed to decode\n");
            return 1;
        }
    }

    for (const auto & [name, n] : state.seq) {
        LOG_INF("abyss-dump: %s x%d\n", name.c_str(), n);
    }

    llama_backend_free();
    return 0;
}
