#include "models.h"
#include "llama-impl.h"

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

// ref: deepseek-ai/DeepSeek-V4.1-Flash-1017 inference/model.py, engram.py
// LLAMA_DSV41_SPARSE=0 disables the sparse attention path (see build_attention41)

static constexpr int64_t DSV41_ENGRAM_DEAD = -1;

static bool dsv41_env_on(const char * name) {
    const char * v = getenv(name);
    return !v || atoi(v) != 0;
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    hparams.n_embd_out_impl = hparams.n_embd; // no hc_head
    if (hparams.indexer_top_k == 0) {
        throw std::runtime_error("DeepSeek-V4.1 indexer top_k must be > 0");
    }

    // each compressed layer reads the KV / top-k of the latest source layer at or before it
    std::vector<uint32_t> kv_src;
    std::vector<uint32_t> idx_src;
    ml.get_arr(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,      kv_src);
    ml.get_arr(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS, idx_src);

    const uint32_t n_layer = hparams.n_layer();
    const auto out_of_range = [&](const std::vector<uint32_t> & v) {
        return std::any_of(v.begin(), v.end(), [&](uint32_t il) { return il >= n_layer; });
    };
    if (out_of_range(kv_src) || out_of_range(idx_src)) {
        throw std::runtime_error("DeepSeek-V4.1 kv/index source layer out of range");
    }
    hparams.dsv41_kv_src.fill(-1);
    hparams.dsv41_topk_src.fill(-1);
    int32_t last_kv  = -1;
    int32_t last_idx = -1;
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t ratio = hparams.dsv4_compress_ratios[il];
        const bool is_kv  = std::find(kv_src.begin(),  kv_src.end(),  il) != kv_src.end();
        const bool is_idx = std::find(idx_src.begin(), idx_src.end(), il) != idx_src.end();
        if ((is_kv || is_idx) && ratio == 0) {
            throw std::runtime_error(format("DeepSeek-V4.1 source layer %u has compress ratio 0", il));
        }
        last_kv  = is_kv  ? (int32_t) il : last_kv;
        last_idx = is_idx ? (int32_t) il : last_idx;
        if (ratio == 0) {
            continue;
        }
        if (ratio > 2 || last_kv < 0 || last_idx < last_kv ||
                hparams.dsv4_compress_ratios[last_kv] != ratio || hparams.dsv4_compress_ratios[last_idx] != ratio) {
            throw std::runtime_error(format("DeepSeek-V4.1 layer %u: ratio %u without a matching kv/index source", il, ratio));
        }
        hparams.dsv41_kv_src[il]   = last_kv;
        hparams.dsv41_topk_src[il] = last_idx;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, hparams.dsv41_cand_src, false);
    if (hparams.dsv41_cand_src >= 0) {
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, hparams.dsv41_cand_topk_blocks);
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   hparams.dsv41_cand_block_size);
        const int32_t cs = hparams.dsv41_cand_src;
        if ((uint32_t) cs >= n_layer || hparams.dsv41_topk_src[cs] != cs || hparams.dsv41_cand_block_size == 0 ||
                hparams.dsv41_cand_topk_blocks == 0) {
            throw std::runtime_error("DeepSeek-V4.1 invalid candidate selection (source must be an index source, block size and top-k blocks > 0)");
        }
        for (uint32_t il = cs + 1; il < n_layer; ++il) {
            if (hparams.dsv41_topk_src[il] == (int32_t) il && hparams.dsv4_compress_ratios[il] != hparams.dsv4_compress_ratios[cs]) {
                throw std::runtime_error("DeepSeek-V4.1 candidate consumers must share the source's compress ratio");
            }
        }
    }

    // Engram: the PLE n-gram hash of qwen4exp with one set of hash constants per Engram layer, stored back to back
    std::vector<uint32_t> e_layers;
    ml.get_arr(LLM_KV_PLE_LAYERS, e_layers, false);
    if (!e_layers.empty()) {
        std::vector<uint64_t> mult;
        std::vector<uint64_t> offsets;
        std::vector<uint64_t> primes;
        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,             hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM,        hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.ple_head_dim);
        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS,      mult);
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,           offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES,       primes);
        ml.get_key(LLM_KV_ENGRAM_PAD_TOKEN_ID,        hparams.engram_pad_id);
        ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,           engram_token_map);

        const size_t n_e    = e_layers.size();
        const size_t n_gram = hparams.ple_ngram_size;
        const size_t n_cols = n_gram >= 2 ? (n_gram - 1)*hparams.ple_heads_per_ngram : 0;
        if (n_cols == 0 || hparams.ple_head_dim == 0 ||
                mult.size() != n_e*n_gram || primes.size() != n_e*n_cols || offsets.size() != n_e*n_cols ||
                mult.size() > LLAMA_MAX_PLE_NGRAM || primes.size() > LLAMA_MAX_PLE_HEADS ||
                hparams.engram_pad_id >= engram_token_map.size()) {
            throw std::runtime_error("DeepSeek-V4.1 engram metadata is inconsistent");
        }
        for (size_t e = 0; e < n_e; ++e) {
            // module e belongs to the e-th Engram layer
            if (e_layers[e] >= n_layer || (e > 0 && e_layers[e] <= e_layers[e - 1])) {
                throw std::runtime_error(format("DeepSeek-V4.1 engram layer %u is out of range or out of order", e_layers[e]));
            }
            hparams.is_ple_impl.set(e_layers[e]);
        }
        for (size_t c = 0; c < n_e*n_cols; ++c) {
            // a bucket count of 0 traps in the hash; row ids are int32
            if (primes[c] < 2 || offsets[c] + primes[c] > (uint64_t) INT32_MAX) {
                throw std::runtime_error(format("DeepSeek-V4.1 engram head %zu: bucket range out of range", c));
            }
            hparams.ple_head_offsets[c]     = (uint32_t) offsets[c];
            hparams.ple_head_vocab_sizes[c] = (uint32_t) primes[c];
        }
        std::copy(mult.begin(), mult.end(), hparams.ple_layer_multipliers.begin());
        hparams.ple_n_heads   = (uint32_t) n_cols;
        hparams.ple_n_modules = (uint32_t) n_e;
    }
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_embd_head     = hparams.n_embd_head_k();
    const int64_t o_groups        = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult         = hparams.dsv4_hc_mult;
    const int64_t hc_dim          = hc_mult * n_embd;
    const int64_t hc_mix_dim      = (2 + hc_mult) * hc_mult;
    const int64_t n_idx_embd      = hparams.indexer_head_size;
    const int64_t n_idx_head      = hparams.indexer_n_head;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // abyss: ABYSS_HEAD_SPLIT_ONLY leaves the head out of the load, llama_model_base::load_tensors reads its two parts
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab},
            llama_abyss_head_split_only() ? TENSOR_SKIP : 0);

    // all Engram tables in one lazily read tensor, like the qwen4exp n-gram table
    if (hparams.ple_n_heads > 0) {
        int64_t n_rows = 0;
        for (size_t c = 0; c < (size_t) hparams.ple_n_modules*hparams.ple_n_heads; ++c) {
            n_rows = std::max(n_rows, (int64_t) hparams.ple_head_offsets[c] + hparams.ple_head_vocab_sizes[c]);
        }
        const std::string name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * w = ml.get_weight(name.c_str())) {
            if (w->tensor->ne[1] < n_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the Engram heads (%" PRId64 ")",
                                                name.c_str(), w->tensor->ne[1], n_rows));
            }
            n_rows = w->tensor->ne[1];
        }
        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"), {(int64_t) hparams.ple_head_dim, n_rows}, TENSOR_READ_LAZY);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        if (hparams.dsv41_kv_src[i] == i) {
            layer.attn_comp_wkv  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,  "weight", i), {n_embd, n_embd_head}, 0);
            layer.attn_comp_norm = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", i), {n_embd_head}, 0);
            if (hparams.dsv4_compress_ratios[i] > 1) {
                layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, 0);
            }
            layer.indexer_attn_k = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", i), {n_embd_head, n_idx_embd}, 0);
            layer.indexer_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {n_idx_embd}, 0);
        }
        if (hparams.dsv41_topk_src[i] == i) {
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, n_idx_head * n_idx_embd}, 0);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_idx_head}, 0);
        }

        if (engram_index(i) >= 0) {
            layer.engram_kv   = create_tensor(tn(LLM_TENSOR_ENGRAM_KV,   "weight", i), {(int64_t) hparams.ple_n_heads * hparams.ple_head_dim, n_embd * (hc_mult + 1)}, 0);
            layer.engram_gate = create_tensor(tn(LLM_TENSOR_ENGRAM_GATE, "weight", i), {n_embd, hc_mult}, 0);
        }

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("DeepSeek-V4.1 has no MTP graph: its DSpark draft is a separate DFLASH GGUF "
                "(convert_hf_to_gguf.py --dspark), used with --spec-type draft-dspark");
    }
    return std::make_unique<graph>(params);
}

// Engram n-gram hash (engram.py NgramHashState.forward)
int llama_model_deepseek41::engram_index(int il) const {
    if (hparams.ple_n_heads == 0 || !hparams.is_ple(il)) {
        return -1;
    }
    int e = 0;
    for (int j = 0; j < il; ++j) {
        e += hparams.is_ple(j);
    }
    return e;
}

void llama_model_deepseek41::engram_hash(int e, const llama_ubatch & ubatch, const std::vector<llama_token> & prev, std::vector<int32_t> & out) const {
    const int64_t n_gram = hparams.ple_ngram_size;
    const int64_t n_prev = n_gram - 1;
    const int64_t n_head = hparams.ple_heads_per_ngram;
    const int64_t n_cols = hparams.ple_n_heads;

    const uint64_t * mult   = hparams.ple_layer_multipliers.data() + (size_t) e*n_gram;
    const uint32_t * primes = hparams.ple_head_vocab_sizes.data()  + (size_t) e*n_cols;
    const int64_t    pad    = engram_token_map[hparams.engram_pad_id];
    const uint32_t * offs   = hparams.ple_head_offsets.data()      + (size_t) e*n_cols;

    const auto cid = [&](llama_token t) -> int64_t {
        return t < 0 || t >= (llama_token) engram_token_map.size() ? DSV41_ENGRAM_DEAD : engram_token_map[t];
    };

    out.resize(ubatch.n_tokens*n_cols);
    std::vector<uint64_t> toks(n_gram);
    for (int64_t i = 0; i < ubatch.n_tokens; ++i) {
        // shift 0 is the token itself; once a shift is blocked (sequence start or dead token) every longer one is too
        const bool is_embd = !ubatch.token || (ubatch.is_mixed() && ubatch.type[i]);
        int64_t c = is_embd ? DSV41_ENGRAM_DEAD : cid(ubatch.token[i]);
        bool blocked = c == DSV41_ENGRAM_DEAD;
        toks[0] = blocked ? pad : c;
        for (int64_t s = 1; s < n_gram; ++s) {
            c = ubatch.pos[i] - s < 0 ? DSV41_ENGRAM_DEAD : cid(prev[i*n_prev + (n_prev - s)]);
            blocked = blocked || c == DSV41_ENGRAM_DEAD;
            toks[s] = blocked ? pad : c;
        }

        uint64_t rolling = toks[0]*mult[0];
        for (int64_t o = 1; o < n_gram; ++o) {
            rolling ^= toks[o]*mult[o];
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t col = (o - 1)*n_head + h;
                out[i*n_cols + col] = (int32_t) (rolling % primes[col] + offs[col]);
            }
        }
    }
}

// cand_pin: +1e30 on the block with the newest compressed row of each query, so a partly filled block is always kept; the reference uses +inf, but a finite pin keeps fully masked blocks at -inf instead of NaN
class llm_graph_input_dsv41 : public llm_graph_input_i {
public:
    llm_graph_input_dsv41(const llama_model_deepseek41 & model, const llama_kv_cache_dsv4_context * mctx) :
        model(model), mctx(mctx) {}

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::vector<ggml_tensor *> engram_ids;
    ggml_tensor * engram_keep = nullptr;    // 0 for embedding (image) tokens; only in mixed ubatches
    ggml_tensor * cand_pin = nullptr;
    bool cand_csa = false;

    const llama_model_deepseek41 & model;
    const llama_kv_cache_dsv4_context * mctx;

private:
    const llama_kv_cache_dsv4_context::comp_plan & cand_plan(const llama_ubatch & ubatch) const {
        return cand_csa ? mctx->get_csa_plan(ubatch) : mctx->get_hca_plan(ubatch);
    }
};

void llm_graph_input_dsv41::set_input(const llama_ubatch * ubatch) {
    if (!engram_ids.empty()) {
        // a token shared by several sequences has the same history in each, so its first sequence resolves it
        std::vector<llama_token> prev;
        mctx->get_raw()->get_prev_tokens(*ubatch, model.hparams.ple_ngram_size - 1, prev);
        // hash all Engram layers first so a lazily read table gets one batched readahead before the first gather
        std::vector<std::vector<int32_t>> ids(engram_ids.size());
        std::vector<int32_t> all;
        for (size_t e = 0; e < engram_ids.size(); ++e) {
            model.engram_hash((int) e, *ubatch, prev, ids[e]);
            all.insert(all.end(), ids[e].begin(), ids[e].end());
        }
        if (model.can_prefetch.count(model.per_layer_tok_embd)) {
            llama_prefetch_rows(model.per_layer_tok_embd, all.data(), all.size());
        }
        for (size_t e = 0; e < engram_ids.size(); ++e) {
            if (engram_ids[e]->buffer) {
                ggml_backend_tensor_set(engram_ids[e], ids[e].data(), 0, ids[e].size()*sizeof(int32_t));
            }
        }
    }

    // embedding (image) tokens are hashed DEAD and get no Engram contribution (gate 0)
    if (engram_keep && engram_keep->buffer) {
        std::vector<float> keep(ubatch->n_tokens);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[i]);
            keep[i] = is_embd ? 0.0f : 1.0f;
        }
        ggml_backend_tensor_set(engram_keep, keep.data(), 0, keep.size()*sizeof(float));
    }

    if (cand_pin && cand_pin->buffer) {
        const auto & plan = cand_plan(*ubatch);
        const int64_t n_blk = cand_pin->ne[0];
        const int64_t bs    = model.hparams.dsv41_cand_block_size;
        GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) ubatch->n_tokens);
        std::vector<float> pin(n_blk*ubatch->n_tokens, 0.0f);
        for (int64_t i = 0; i < (int64_t) ubatch->n_tokens; ++i) {
            const int64_t nv = plan.n_visible[i];
            if (nv > 0) {
                GGML_ASSERT((nv - 1)/bs < n_blk);
                pin[i*n_blk + (nv - 1)/bs] = 1e30f;
            }
        }
        ggml_backend_tensor_set(cand_pin, pin.data(), 0, pin.size()*sizeof(float));
    }
}

bool llm_graph_input_dsv41::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    bool res = true;
    for (auto * t : engram_ids) {
        res &= t->ne[0] == (int64_t) model.hparams.ple_n_heads*params.ubatch.n_tokens;
    }
    res &= (engram_keep != nullptr) == (!engram_ids.empty() && params.ubatch.embd != nullptr);
    if (engram_keep) {
        res &= engram_keep->ne[2] == (int64_t) params.ubatch.n_tokens;
    }
    if (cand_pin) {
        const auto & plan = cand_plan(params.ubatch);
        const int64_t bs = model.hparams.dsv41_cand_block_size;
        res &= plan.n_kv > 0 && cand_pin->ne[0] == (plan.n_kv + bs - 1)/bs;
        res &= cand_pin->ne[1]*cand_pin->ne[3] == (int64_t) params.ubatch.n_tokens;
    }
    return res;
}

ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model_deepseek41 & model,
        ggml_tensor * x,
        ggml_tensor * rows,
        ggml_tensor * keep,
        int il) const {
    const auto & layer = model.layers[il];
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    GGML_ASSERT(rows->ne[0] == (int64_t) (hparams.ple_head_dim*hparams.ple_n_heads) && rows->ne[1] == nt);

    ggml_tensor * kv = build_lora_mm(layer.engram_kv, rows);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, hc, nt, ggml_row_size(kv->type, n_embd), kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, 1,  nt, ggml_row_size(kv->type, n_embd), kv->nb[1],
            ggml_row_size(kv->type, hc*n_embd)));

    // quantized copy has no binary-op kernel
    ggml_tensor * egate = ggml_is_quantized(layer.engram_gate->type) ? ggml_cast(ctx0, layer.engram_gate, GGML_TYPE_F32) : layer.engram_gate;
    ggml_tensor * dot = ggml_mul(ctx0, ggml_mul(ctx0, ggml_rms_norm(ctx0, x, norm_rms_eps), ggml_rms_norm(ctx0, key, norm_rms_eps)), egate);
    dot = ggml_scale(ctx0, ggml_sum_rows(ctx0, dot), 1.0f/sqrtf((float) n_embd));
    cb(dot, "engram_dot", il);

    // sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)); sign(0) = +1 as in copysign
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_neg(ctx0, dot)), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, mag, sign));
    if (keep) {
        gate = ggml_mul(ctx0, gate, keep);
    }
    cb(gate, "engram_gate", il);

    return ggml_add(ctx0, x, ggml_mul(ctx0, ggml_repeat_4d(ctx0, value, n_embd, hc, nt, 1), gate));
}

// abyss: rows [r0, r0 + n) of a graph input with a row per token of the ubatch, [ne0, n_tokens, 1, 1], one view per range
// for all the layers
ggml_tensor * llama_model_deepseek41::graph::row_view(ggml_tensor * t, int64_t r0, int64_t n) {
    GGML_ASSERT(t->ne[2] == 1 && t->ne[3] == 1 && r0 >= 0 && n > 0 && r0 + n <= t->ne[1]);
    if (r0 == 0 && n == t->ne[1]) {
        return t;
    }
    ggml_tensor *& view = row_views[{ t, r0, n }];
    if (view == nullptr) {
        view = ggml_view_4d(ctx0, t, t->ne[0], n, 1, 1, t->nb[1], n*t->nb[1], n*t->nb[1], r0*t->nb[1]);
    }
    return view;
}

ggml_tensor * llama_model_deepseek41::graph::build_attention41(
        const llama_model_deepseek41 & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * cand_pin,
        int il) {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t n_idx_embd       = hparams.indexer_head_size;
    const int64_t n_idx_head       = hparams.indexer_n_head;
    const int64_t nt               = cur->ne[1];
    const float   kq_scale         = 1.0f/sqrtf(float(n_embd_head));
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    const bool    comp_rope     = ratio != 0;
    const float   freq_base_l   = comp_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float   freq_scale_l  = comp_rope ? freq_scale : 1.0f;
    const float   ext_factor_l  = comp_rope ? ext_factor : 0.0f;
    const float   attn_factor_l = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float   beta_fast_l   = comp_rope ? beta_fast : 0.0f;
    const float   beta_slow_l   = comp_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l  = comp_rope ? n_ctx_orig : 0;

    const auto rope = [&](ggml_tensor * t, ggml_tensor * pos, int64_t n_nope) {
        t = ggml_rope_ext(ctx0, t, pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        return ggml_rope_set_offset(t, n_nope);
    };

    const auto & ci         = ratio == 2 ? inp_dsv4->get_csa()             : inp_dsv4->get_hca();
    const auto * comp_ctx   = ratio == 2 ? inp_dsv4->mctx->get_csa()       : inp_dsv4->mctx->get_hca();
    const auto * comp_state = ratio == 2 ? inp_dsv4->mctx->get_csa_state() : inp_dsv4->mctx->get_hca_state();

    // abyss: under the CED-aware prefill a layer attends only for the last nq tokens of the ubatch, see
    // llama_context::ced_plan(). Every token still writes its KV. The window-only path keeps every query
    const int64_t nq = !ced_n_sub.empty() && ratio != 0 && ci.kq_mask ? ced_n_sub[il] : nt;
    GGML_ASSERT(nq >= 0 && nq <= nt);

    const bool sub = nq < nt;

    // rows [r0, r0 + n) of t [ne0, rows, 1, 1]
    const auto rows_at = [&](ggml_tensor * t, int64_t r0, int64_t n) {
        GGML_ASSERT(t->ne[2] == 1 && t->ne[3] == 1 && r0 >= 0 && n > 0 && r0 + n <= t->ne[1]);
        return r0 == 0 && n == t->ne[1] ? t :
            ggml_view_4d(ctx0, t, t->ne[0], n, 1, 1, t->nb[1], n*t->nb[1], n*t->nb[1], r0*t->nb[1]);
    };

    ggml_tensor * cur_q = nq > 0 ? rows_at(cur, nt - nq, nq) : nullptr;
    ggml_tensor * pos_q = sub && nq > 0 ? ggml_view_1d(ctx0, inp_pos, nq, (nt - nq)*inp_pos->nb[0]) : inp_pos;

    // with no queries they stay out of the graph: expanded, they would be computed and kept to the end of the graph
    ggml_tensor * qr = nullptr;
    ggml_tensor * q  = nullptr;
    if (nq > 0) {
        qr = build_norm(build_lora_mm(layer.wq_a, cur_q), layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(qr, "qr_norm", il);

        q = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq_b, qr), n_embd_head, n_head, nq);
        q = rope(q, pos_q, n_embd_head_nope);
        cb(q, "q", il);
    }

    ggml_tensor * kv = build_norm(build_lora_mm(layer.wkv, cur), layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = rope(ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt), inp_pos, n_embd_head_nope);
    cb(kv, "kv", il);

    const int32_t rd = inp_dsv4->mctx->get_comp_read_layer(il);

    if (ratio != 0) {
        GGML_ASSERT(hparams.dsv41_kv_src[il] >= 0 && (int64_t) comp_state->get_ratio() == ratio);

        if (hparams.dsv41_kv_src[il] == il && ci.state_pos) {
            ggml_tensor * state_kv    = build_lora_mm(layer.attn_comp_wkv, cur);
            ggml_tensor * state_score = ratio > 1 ? build_lora_mm(layer.attn_comp_wgate, cur) : ggml_fill(ctx0, state_kv, 0.0f);

            ggml_tensor * restored_kv    = comp_state->get_kv_all(ctx0, il);
            ggml_tensor * restored_score = comp_state->get_score_all(ctx0, il);
            if (ci.state_restore_src_idxs && ci.state_restore_dst_idxs) {
                restored_kv    = comp_state->cpy_kv   (ctx0, ggml_get_rows(ctx0, restored_kv,    ci.state_restore_src_idxs), ci.state_restore_dst_idxs, il);
                restored_score = comp_state->cpy_score(ctx0, ggml_get_rows(ctx0, restored_score, ci.state_restore_src_idxs), ci.state_restore_dst_idxs, il);
            }

            if (ci.state_write_idxs) {
                const int64_t n_blocks = ci.state_write_pos->ne[0];
                const int64_t n_state  = comp_state->get_n_rows();
                GGML_ASSERT(ci.state_read_idxs->ne[0] == ratio*n_blocks);

                ggml_tensor * src_kv    = ggml_concat(ctx0, ggml_view_2d(ctx0, restored_kv,    restored_kv->ne[0],    n_state, restored_kv->nb[1],    0), state_kv,    1);
                ggml_tensor * src_score = ggml_concat(ctx0, ggml_view_2d(ctx0, restored_score, restored_score->ne[0], n_state, restored_score->nb[1], 0), state_score, 1);

                ggml_tensor * v = ggml_get_rows(ctx0, src_kv,    ci.state_read_idxs);
                ggml_tensor * s = ggml_get_rows(ctx0, src_score, ci.state_read_idxs);
                v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, v, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));
                s = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, s, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));

                ggml_tensor * comp = ggml_sum_rows(ctx0, ggml_mul(ctx0, v, ggml_soft_max(ctx0, s)));
                comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
                comp = build_norm(comp, layer.attn_comp_norm, nullptr, LLM_NORM_RMS, il);
                cb(comp, "comp_latent", il);

                // index K from the RoPE-free latent; a latent stands for the first token of its group
                ggml_tensor * ik = build_norm(build_lora_mm(layer.indexer_attn_k, comp), layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
                ik = rope(ik, ci.state_write_pos, n_idx_embd - n_embd_head_rope);
                comp = rope(comp, ci.state_write_pos, n_embd_head_nope);

                ggml_tensor * row = ggml_concat(ctx0, comp, ik, 0);
                ggml_build_forward_expand(gf, comp_ctx->cpy_k(ctx0, row, ci.state_write_idxs, il));
                comp_row_of[il] = row;
            }

            if (ci.state_snapshot_src_idxs && ci.state_snapshot_dst_idxs) {
                ggml_tensor * snap_kv    = ggml_concat(ctx0, restored_kv,    state_kv,    1);
                ggml_tensor * snap_score = ggml_concat(ctx0, restored_score, state_score, 1);
                ggml_build_forward_expand(gf, comp_state->cpy_kv   (ctx0, ggml_get_rows(ctx0, snap_kv,    ci.state_snapshot_src_idxs), ci.state_snapshot_dst_idxs, il));
                ggml_build_forward_expand(gf, comp_state->cpy_score(ctx0, ggml_get_rows(ctx0, snap_score, ci.state_snapshot_src_idxs), ci.state_snapshot_dst_idxs, il));
            }
            ggml_build_forward_expand(gf, comp_state->cpy_kv   (ctx0, ggml_get_rows(ctx0, state_kv,    ci.state_persist_src_idxs), ci.state_persist_dst_idxs, il));
            ggml_build_forward_expand(gf, comp_state->cpy_score(ctx0, ggml_get_rows(ctx0, state_score, ci.state_persist_src_idxs), ci.state_persist_dst_idxs, il));
        }

        if (rd == il && hparams.dsv41_kv_src[il] != il && comp_row_of[hparams.dsv41_kv_src[il]]) {
            // first consumer on this device: append the source's new rows to the local mirror
            ggml_build_forward_expand(gf, comp_ctx->cpy_k(ctx0, comp_row_of[hparams.dsv41_kv_src[il]], ci.state_write_idxs, il));
        }
    }

    ggml_tensor * out = nullptr;
    if (ratio == 0 || !ci.kq_mask) {
        // no compressed row visible yet: window only
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);
    } else {
        const int64_t n_comp = ci.kq_mask->ne[0];

        ggml_tensor * comp_all = comp_ctx->get_k(ctx0, rd);
        GGML_ASSERT(comp_all->ne[0] == n_embd_head + n_idx_embd && n_comp <= comp_all->ne[2]);
        const int64_t n_stream = comp_all->ne[3];

        // a quantized K cache stores the window Hadamard-rotated (as build_raw_attention)
        ggml_tensor * k_rot = inp_attn->self_k_rot[il];

        if (nq == 0) {
            // abyss: no token attends here, the layer only writes its window KV
            if (k_rot) {
                kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
            }
            ggml_build_forward_expand(gf, kv);
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

            return ggml_fill(ctx0, cur, 0.0f);
        }

        // abyss: the plan covers one sequence, so a layer that attends for part of the ubatch has one stream
        GGML_ASSERT(!sub || n_stream == 1);

        // abyss: ABYSS_DSV41_CHUNK=N scores, masks and attends the queries of a layer in chunks of at most N, so that the
        // scores and the masks, a row per compressed position and query, exist for one chunk at a time. The chunks
        // are cut from the end of the ubatch, and the rows of an input are one view for all the layers (row_view): a
        // view of a graph input stays on the host and the scheduler uploads it once per split that reads it, whatever
        // the number of layers in the split. A layer that attends for the last nq tokens only (CED) starts with a
        // shorter chunk of its own. Without chunks and with every query, the inputs are read as they are
        static const int64_t chunk_env = [] {
            const char * e = getenv("ABYSS_DSV41_CHUNK");
            return e != nullptr ? std::max<int64_t>(0, atoll(e)) : 0;
        }();
        struct span {
            int64_t r0; // first row in the ubatch
            int64_t n;
        };
        std::vector<span> spans;
        const int64_t q0 = nt - nq;
        if (n_stream == 1 && (sub || (chunk_env > 0 && nq > chunk_env))) {
            const int64_t step = chunk_env > 0 ? chunk_env : nq;
            for (int64_t end = nt; end > q0; ) {
                const int64_t r0 = std::max(q0, nt - ((nt - end)/step + 1)*step);
                spans.insert(spans.begin(), { r0, end - r0 });
                end = r0;
            }
        }
        const bool whole = spans.empty();

        // the top-k and the candidate blocks of a source layer cover its own queries, which include this layer's
        ggml_tensor * top_k = topk_of[hparams.dsv41_topk_src[il]];
        if (hparams.dsv41_topk_src[il] == il) {
            ggml_tensor * ik = ggml_view_4d(ctx0, comp_all, n_idx_embd, 1, n_comp, n_stream,
                    comp_all->nb[1], comp_all->nb[2], comp_all->nb[3], ggml_row_size(comp_all->type, n_embd_head));
            // the fused indexer reads the strided view in place, the unfused path needs it contiguous; CUDA has no same-type copy of quantized views, so dequantize them
            const bool fused_idx = sparse && cparams.fused_lid && ci.kq_mask->type == GGML_TYPE_F16;
            if (ggml_is_quantized(ik->type)) {
                ik = ggml_cast(ctx0, ik, GGML_TYPE_F32);
            } else if (!fused_idx) {
                ik = ggml_cont(ctx0, ik);
            }

            ggml_tensor * iq = ggml_reshape_3d(ctx0, build_lora_mm(layer.indexer_attn_q_b, qr), n_idx_embd, n_idx_head, nq);
            iq = rope(iq, pos_q, n_idx_embd - n_embd_head_rope);
            iq = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nq/n_stream, n_stream, iq->nb[1], iq->nb[2], iq->nb[2]*(nq/n_stream), 0);

            ggml_tensor * iw = ggml_scale(ctx0, build_lora_mm(layer.indexer_proj, cur_q), 1.0f/sqrtf(float(n_idx_embd*n_idx_head)));
            iw = ggml_view_4d(ctx0, iw, iw->ne[0], nq/n_stream, 1, n_stream, iw->nb[1], iw->nb[1]*(nq/n_stream), iw->nb[1]*(nq/n_stream), 0);

            // chunk queries so the [n_ih, tokens, n_comp] scores stay near 2^26 elements (at most 64 chunks); the fused indexer runs as one chunk
            // abyss: or as the chunks above
            const int64_t n_tok = nq/n_stream;
            GGML_ASSERT(whole ? ci.kq_mask->ne[1] == n_tok : ci.kq_mask->ne[1] == nt);
            const int64_t chunk = fused_idx ? n_tok :
                std::max({ (n_tok + 63)/64, (INT64_C(1) << 26)/(n_idx_head*n_comp*n_stream), INT64_C(1) });
            std::vector<span> idx_chunks; // rows among this layer's queries
            if (fused_idx && !whole) {
                for (const span & sp : spans) {
                    idx_chunks.push_back({ sp.r0 - q0, sp.n });
                }
            } else {
                for (int64_t t0 = 0; t0 < n_tok; t0 += chunk) {
                    idx_chunks.push_back({ t0, std::min(chunk, n_tok - t0) });
                }
            }

            const int32_t cand_src = hparams.dsv41_cand_src;
            const bool    pick_blk = cand_src == il && cand_pin;              // level one: pick the candidate blocks
            const bool    use_blk  = cand_src >= 0 && cand_src < il && cand_sel; // level two: score inside them
            const int64_t bs       = hparams.dsv41_cand_block_size;
            const int64_t n_blk    = pick_blk || use_blk ? (n_comp + bs - 1)/bs : 0;
            GGML_ASSERT(!pick_blk || cand_pin->ne[0] == n_blk);
            GGML_ASSERT(!use_blk  || cand_n_comp == n_comp);

            // abyss: the candidate blocks of the source cover its queries, the last of which are this layer's
            const int64_t sel_off = use_blk ? cand_sel->ne[1] - n_tok : 0;
            GGML_ASSERT(sel_off >= 0);

            // abyss: a constant tensor, repeated from one element. Filling a new tensor instead makes it a leaf of the graph,
            // which the allocator places before the first node and keeps to the end: 192 MiB per index layer with
            // candidate blocks at 196k positions and 2,048 queries
            const auto filled = [&](float value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
                return ggml_repeat_4d(ctx0, ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1), value), ne0, ne1, ne2, ne3);
            };

            ggml_tensor * ik_p = fused_idx ? nullptr : ggml_permute(ctx0, ik, 0, 2, 1, 3);
            std::vector<ggml_tensor *> parts_top_k;
            std::vector<ggml_tensor *> parts_sel;
            for (const span & ic : idx_chunks) {
                const int64_t t0 = ic.r0;
                const int64_t nc = ic.n;
                const bool    all = t0 == 0 && nc == n_tok; // every query of the layer in one chunk
                // rows [off + t0, off + t0 + nc) of a tensor of this graph, per stream
                const auto rows = [&](ggml_tensor * t, int64_t off = 0) {
                    return ggml_view_4d(ctx0, t, t->ne[0], nc, 1, n_stream, t->nb[1], t->nb[2], t->nb[3], (off + t0)*t->nb[1]);
                };
                // the same rows of a graph input, which has a row per token of the ubatch
                const auto rows_inp = [&](ggml_tensor * t) {
                    return whole ? (all ? t : rows(t)) : row_view(t, q0 + t0, nc);
                };
                ggml_tensor * iqc = all ? iq :
                    ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nc, n_stream, iq->nb[1], iq->nb[2], iq->nb[3], t0*iq->nb[2]);
                ggml_tensor * score = nullptr;
                if (fused_idx) {
                    score = ggml_lightning_indexer(ctx0, iqc, ik, all ? iw : rows(iw), rows_inp(ci.kq_mask));
                    res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
                } else {
                    ggml_tensor * kq = ggml_mul_mat(ctx0, ik_p, ggml_permute(ctx0, iqc, 0, 2, 1, 3));
                    kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
                    score = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_relu(ctx0, kq), rows(iw)));
                    ggml_tensor * mask  = whole ? rows(ci.kq_mask) : row_view(ci.kq_mask, q0 + t0, nc);
                    score = ggml_add(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)),
                            mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32));
                }
                cb(score, "idx_score", il);

                if (pick_blk) {
                    // best position per block, pin the newest block, keep the top blocks
                    ggml_tensor * s2 = ggml_reshape_2d(ctx0, score, n_comp, nc*n_stream);
                    if (n_blk*bs != n_comp) {
                        s2 = ggml_concat(ctx0, s2, filled(-INFINITY, n_blk*bs - n_comp, nc*n_stream, 1, 1), 0);
                    }
                    ggml_tensor * bmax = ggml_pool_1d(ctx0, s2, GGML_OP_POOL_MAX, bs, bs, 0);
                    bmax = ggml_add(ctx0, ggml_reshape_4d(ctx0, bmax, n_blk, nc, 1, n_stream), whole ? rows(cand_pin) : row_view(cand_pin, q0 + t0, nc));
                    cb(bmax, "cand_block_score", il);
                    parts_sel.push_back(ggml_top_k(ctx0, bmax, std::min<int64_t>(hparams.dsv41_cand_topk_blocks, n_blk)));
                } else if (use_blk) {
                    // own scores, restricted to the source's candidate blocks: 0 inside them, -inf elsewhere
                    ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel, sel_off));
                    const int64_t k_b = bsel->ne[0];
                    ggml_tensor * bm    = filled(-INFINITY, 1, n_blk, nc*n_stream, 1);
                    ggml_tensor * zeros = filled(0.0f,      1, k_b,   nc*n_stream, 1);
                    bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc*n_stream, 1));
                    bm = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, bm, bs, n_blk, nc*n_stream, 1), bs*n_blk, nc*n_stream);
                    bm = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, bm, n_comp, nc*n_stream, bm->nb[1], 0)), n_comp, nc, 1, n_stream);
                    cb(bm, "cand_mask", il);
                    score = ggml_add(ctx0, score, bm);
                }

                parts_top_k.push_back(ggml_top_k(ctx0, score, std::min<int64_t>(n_comp, hparams.indexer_top_k)));
                if (!whole) {
                    // a chunk at a time: its scores are freed before the next ones are computed
                    ggml_build_forward_expand(gf, parts_top_k.back());
                    if (pick_blk) {
                        ggml_build_forward_expand(gf, parts_sel.back());
                    }
                }
            }

            const auto join = [&](std::vector<ggml_tensor *> v) {
                while (v.size() > 1) {
                    std::vector<ggml_tensor *> w;
                    for (size_t i = 0; i < v.size(); i += 2) {
                        w.push_back(i + 1 < v.size() ? ggml_concat(ctx0, v[i], v[i + 1], 1) : v[i]);
                    }
                    v.swap(w);
                }
                return v[0];
            };
            if (pick_blk) {
                cand_sel    = join(parts_sel);
                cand_n_comp = n_comp;
            }
            top_k = ggml_cont(ctx0, join(parts_top_k));
            cb(top_k, "idx_top_k", il);
            topk_of[il] = top_k;
        }
        GGML_ASSERT(top_k && "DeepSeek-V4.1: top-k source layer produced no top-k");

        if (k_rot) {
            q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
            kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
        }
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, kv);
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

        // abyss: only the rows of the window when the mask is over their list (ABYSS_DSV41_WINDOW_ROWS)
        ggml_tensor * raw_k    = inp_attn->window_k(ctx0, inp_attn->mctx->get_k(ctx0, il));
        ggml_tensor * raw_mask = inp_attn->self_attn_mask.raw;

        const int32_t topk_src = hparams.dsv41_topk_src[il];
        const int64_t k_sel    = top_k->ne[0];
        const int64_t n_raw    = raw_k->ne[2];

        // few queries per stream: attend over [window | top-k rows of the query], one FA stream per query; same keys and mask as dense, different key order
        // abyss: the gathered rows are shared by the layers of a top-k source, so a layer with its own subset of queries attends densely
        const bool gather = sparse && cparams.flash_attn && raw_mask->type == GGML_TYPE_F16 && ci.kq_mask->type == GGML_TYPE_F16 && whole &&
                ((n_stream == 1 && nt <= 32) || nt == n_stream);

        if (gather) {
            // ggml_top_k returns the rows in no fixed order and FA sums the keys in that order: sort them so the result is deterministic
            ggml_tensor *& sel = gather_sel_of[topk_src];
            if (!sel) {
                GGML_ASSERT(n_comp <= (INT64_C(1) << 24)); // row ids are exact in F32
                ggml_tensor * ids = ggml_reshape_2d(ctx0, top_k, k_sel, nt);
                ggml_tensor * ord = ggml_argsort(ctx0, ggml_cast(ctx0, ids, GGML_TYPE_F32), GGML_SORT_ORDER_ASC);
                sel = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, ids, 1, k_sel, nt), ord), k_sel, nt);
                cb(sel, "comp_top_k_sorted", il);
            }

            // built once per (storage, top-k) pair
            ggml_tensor *& comp_k = gather_k_of[{rd, topk_src}];
            if (!comp_k) {
                if (n_stream == 1) {
                    ggml_tensor * src_rows = ggml_view_3d(ctx0, comp_all, n_embd_head, n_comp, 1, comp_all->nb[2], comp_all->nb[3], 0);
                    comp_k = ggml_get_rows(ctx0, src_rows, ggml_reshape_1d(ctx0, sel, k_sel*nt));
                } else {
                    ggml_tensor * src_rows = ggml_view_3d(ctx0, comp_all, n_embd_head, n_comp, n_stream, comp_all->nb[2], comp_all->nb[3], 0);
                    comp_k = ggml_get_rows(ctx0, src_rows, sel);
                }
                comp_k = ggml_reshape_4d(ctx0, comp_k, n_embd_head, 1, k_sel, nt); // F32 (get_rows dequantizes)
                if (k_rot) {
                    // quantized K cache: the window is stored rotated, the compressed rows are not
                    comp_k = llama_mul_mat_hadamard(ctx0, comp_k, k_rot);
                }
                if (comp_k->type != raw_k->type) {
                    comp_k = ggml_cast(ctx0, comp_k, raw_k->type); // exact: rows are raw_k->type values
                }
                cb(comp_k, "comp_k_sel", il);
            }

            if (n_stream == 1 && nt > 1) {
                if (ggml_is_quantized(raw_k->type)) {
                    // no REPEAT kernel for quantized types: byte-exact copies by doubling concats
                    ggml_tensor * rep = nullptr;
                    ggml_tensor * pw  = raw_k;
                    for (int64_t n = nt; n > 0; n >>= 1) {
                        if (n & 1) {
                            rep = rep ? ggml_concat(ctx0, rep, pw, 3) : pw;
                        }
                        if (n > 1) {
                            pw = ggml_concat(ctx0, pw, pw, 3);
                        }
                    }
                    raw_k = rep;
                } else {
                    raw_k = ggml_repeat_4d(ctx0, raw_k, n_embd_head, 1, n_raw, nt);
                }
            }
            ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);

            // window mask, then the comp mask at the selected rows (-inf where top-k picked an invisible row); one per top-k source
            ggml_tensor *& kq_mask = gather_mask_of[topk_src];
            if (!kq_mask) {
                ggml_tensor * sel_mask = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, ci.kq_mask, 1, n_comp, nt), sel);
                sel_mask = ggml_cast(ctx0, ggml_reshape_4d(ctx0, sel_mask, k_sel, 1, 1, nt), GGML_TYPE_F16);
                kq_mask = ggml_concat(ctx0, ggml_reshape_4d(ctx0, raw_mask, raw_mask->ne[0], 1, 1, nt), sel_mask, 0);
                cb(kq_mask, "comp_kq_mask_sel", il);
            }

            out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, nullptr, layer.attn_sinks, nullptr, 0, kq_scale, il);
        } else {
            // dense attention over [window | all compressed rows], selection applied as a mask
            ggml_tensor * comp_k = ggml_view_4d(ctx0, comp_all, n_embd_head, 1, n_comp, n_stream,
                    comp_all->nb[1], comp_all->nb[2], comp_all->nb[3], 0);
            if (k_rot) {
                // compressed rows are stored unrotated: rotate in F32, then cast to the cache type
                comp_k = comp_k->type != GGML_TYPE_F32 ? ggml_cast(ctx0, comp_k, GGML_TYPE_F32) : ggml_cont(ctx0, comp_k);
                comp_k = llama_mul_mat_hadamard(ctx0, comp_k, k_rot);
                if (comp_k->type != raw_k->type) {
                    comp_k = ggml_cast(ctx0, comp_k, raw_k->type);
                }
            } else if (ggml_is_quantized(comp_k->type)) {
                // quantized cache without rotation (LLAMA_ATTN_ROT_DISABLE=1): via F32 for the same reason
                comp_k = ggml_cast(ctx0, ggml_cast(ctx0, comp_k, GGML_TYPE_F32), raw_k->type);
            } else {
                comp_k = comp_k->type != raw_k->type ? ggml_cast(ctx0, comp_k, raw_k->type) : ggml_cont(ctx0, comp_k);
            }
            cb(comp_k, "comp_k", il);
            ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);

            // at most window + top-k visible columns per query: lets CUDA FA gather only those
            const int64_t n_kv_max = sparse ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + k_sel : 0;

            if (whole) {
                ggml_tensor * sel_mask = build_top_k_mask(ci.kq_mask, top_k, "comp_top_k_mask", il);
                ggml_tensor * kq_mask  = ggml_concat(ctx0, raw_mask, sel_mask, 0);
                cb(kq_mask, "comp_kq_mask", il);

                out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, nullptr, layer.attn_sinks, nullptr, n_kv_max, kq_scale, il);
            } else {
                // abyss: a chunk of the queries at a time, each with the rows of the masks and of the top-k that are its own
                const int64_t top_off = top_k->ne[1] - nq; // the top-k of a source layer starts at its first query
                GGML_ASSERT(top_off >= 0);
                for (const span & sp : spans) {
                    const int64_t t0 = sp.r0 - q0;
                    ggml_tensor * q_c = sp.n == nq ? q :
                        ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], sp.n, q->nb[1], q->nb[2], t0*q->nb[2]);

                    ggml_tensor * sel_mask = build_top_k_mask(row_view(ci.kq_mask, sp.r0, sp.n), rows_at(top_k, top_off + t0, sp.n), "comp_top_k_mask", il);
                    ggml_tensor * kq_mask  = ggml_concat(ctx0, row_view(raw_mask, sp.r0, sp.n), sel_mask, 0);
                    cb(kq_mask, "comp_kq_mask", il);

                    // build_attn_mha() adds its result to the graph, so the masks of a chunk are freed before the next
                    ggml_tensor * out_c = build_attn_mha(q_c, k_all, k_all, nullptr, kq_mask, nullptr, layer.attn_sinks, nullptr, n_kv_max, kq_scale, il);
                    out = out ? ggml_concat(ctx0, out, out_c, 1) : out_c;
                }
            }
        }
        if (k_rot) {
            out = llama_mul_mat_hadamard(ctx0, out, k_rot);
        }
        cb(out, "attn_comp", il);
    }

    // abyss: the queries keep only their own rows through the output projection
    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nq);
    out = ggml_rope_ext_back(ctx0, out, pos_q, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_permute(ctx0, ggml_reshape_3d(ctx0, out, n_head/n_groups*n_embd_head, n_groups, nq), 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_cont_2d(ctx0, ggml_permute(ctx0, oa, 0, 2, 1, 3), o_lora_rank*n_groups, nq);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    if (sub) {
        // the tokens that do not attend get a zero attention output
        out = ggml_pad_ext(ctx0, out, 0, 0, (int) (nt - nq), 0, 0, 0, 0, 0);
        cb(out, "attn_out_ced", il);
    }

    return out;
}

llama_model_deepseek41::graph::graph(const llm_graph_params & params) :
    llama_model_deepseek4::graph(params, no_build{}) {
    const auto & model = static_cast<const llama_model_deepseek41 &>(this->model);

    static const bool sparse_env = dsv41_env_on("LLAMA_DSV41_SPARSE");
    sparse = sparse_env;

    topk_of.assign(n_layer, nullptr);
    comp_row_of.assign(n_layer, nullptr);

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    ggml_build_forward_expand(gf, inp_dsv4->get_raw()->self_attn_mask.raw);

    auto inp41 = std::make_unique<llm_graph_input_dsv41>(model, inp_dsv4->mctx);
    // embedding input has no token ids: it skips Engram, like the reference's masked (image) positions
    // the gathers go right after the token embeddings, so they share the first graph split (the table stays on the host)
    std::vector<ggml_tensor *> engram_rows;
    for (int il = 0; ubatch.token && hparams.ple_n_heads > 0 && il < n_layer; ++il) {
        if (!hparams.is_ple(il)) {
            continue;
        }
        ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) hparams.ple_n_heads*n_tokens);
        ggml_set_name(ids, ("engram_ids-" + std::to_string(il)).c_str());
        ggml_set_input(ids);
        inp41->engram_ids.push_back(ids);

        ggml_tensor * rows = ggml_get_rows(ctx0, model.per_layer_tok_embd, ids);
        rows = ggml_reshape_2d(ctx0, rows, hparams.ple_head_dim*hparams.ple_n_heads, n_tokens);
        cb(rows, "engram_rows", il);
        ggml_build_forward_expand(gf, rows);
        engram_rows.push_back(rows);
    }
    ggml_tensor * engram_keep = nullptr;
    if (hparams.ple_n_heads > 0 && ubatch.token && ubatch.embd) {
        engram_keep = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
        ggml_set_input(engram_keep);
        ggml_set_name(engram_keep, "engram_keep");
        inp41->engram_keep = engram_keep;
    }
    ggml_tensor * cand_pin = nullptr;
    if (hparams.dsv41_cand_src >= 0) {
        inp41->cand_csa = hparams.dsv4_compress_ratios[hparams.dsv41_cand_src] == 2;
        const auto & ci = inp41->cand_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();
        if (ci.kq_mask) {
            const int64_t bs = hparams.dsv41_cand_block_size;
            cand_pin = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, (ci.kq_mask->ne[0] + bs - 1)/bs, ci.kq_mask->ne[1], 1, ci.kq_mask->ne[3]);
            ggml_set_input(cand_pin);
            ggml_set_name(cand_pin, "cand_pin");
            inp41->cand_pin = cand_pin;
        }
    }
    res->add_input(std::move(inp41));

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // single-pass mHC: the pre-mix used to collapse the stream lags one sublayer
    ggml_tensor * pre_prev = nullptr;

    // abyss: expert deferral, ABYSS_MOE_DEFER=1. In single-token decoding, the experts that the MoE cache leaves to the
    // CPU are not merged into the FFN output of their layer. Their weighted sum D enters the residual one sublayer
    // later, before the post-mix of the next layer's attention, as residual + post ⊗ D with the post-mix of their own
    // FFN: the hidden state is exact from there on, and only the next layer's attention (and the mixes computed from
    // its input) did not see D. The GPU computes the cached experts and that attention while the CPU computes the
    // misses, instead of waiting for them. ABYSS_MOE_DEFER=2 adds D right after its own FFN, which is the undeferred
    // result with another order of operations, to check the plumbing.
    static const int moe_defer = [] {
        const char * e = getenv("ABYSS_MOE_DEFER");
        return e != nullptr ? atoi(e) : 0;
    }();
    const bool defer_ok = moe_defer > 0 && moe_cache != nullptr && n_tokens == 1 && ced_n_sub.empty() &&
        std::none_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(), [](bool tap) { return tap; });
    ggml_tensor * defer_sum  = nullptr; // D of the previous layer, [n_embd, n_tokens]
    ggml_tensor * defer_post = nullptr; // the post-mix of its FFN

    // residual + post ⊗ x
    const auto hc_add = [&](ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post) {
        if (cparams.fused_dsv4_hc_post) {
            return ggml_dsv4_hc_post(ctx0, x, residual, post, nullptr);
        }
        ggml_tensor * d = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
        return ggml_add(ctx0, residual, ggml_mul(ctx0, d, ggml_reshape_3d(ctx0, post, 1, hc, n_tokens)));
    };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int e = model.engram_index(il);
        if (e >= 0 && !engram_rows.empty()) {
            inpL = build_engram(model, inpL, engram_rows[e], engram_keep, il);
            cb(inpL, "engram_out", il);
        }

        // DSpark target features: mean over the hc copies of the layer input
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = build_hc_mean(inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        const hc_mixes attn_mix = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);

        // the first layer's attention collapses with the identity pre-mix: copy 0
        cur = pre_prev ? build_hc_pre(inpL, pre_prev, il) : ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[2], 0));
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention41(model, inp_dsv4, cur, inp_pos, cand_pin, il);

        if (defer_sum != nullptr) {
            // abyss: everything up to here goes into the graph before the misses of the previous layer, so that the
            // scheduler runs it on the GPU while the CPU computes them
            ggml_build_forward_expand(gf, cur);
            ggml_build_forward_expand(gf, attn_mix.post);
            ggml_build_forward_expand(gf, attn_mix.comb);
            residual = hc_add(defer_sum, residual, defer_post);
            cb(residual, "hc_deferred", il);
            defer_sum  = nullptr;
            defer_post = nullptr;
        }

        inpL = build_hc_post(cur, residual, attn_mix.post, attn_mix.comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        const hc_mixes ffn_mix = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        pre_prev = ffn_mix.pre;

        // abyss: under the CED-aware prefill this layer runs its FFN on the last n_ffn tokens of the ubatch only, the
        // tokens that nothing reads anymore get a zero FFN output, see llama_context::ced_plan()
        const int64_t n_ffn = ced_n_sub.empty() ? n_tokens : ced_n_sub[il];
        GGML_ASSERT(n_ffn >= 0 && n_ffn <= n_tokens);

        ggml_tensor * attn_out = cur;
        ggml_tensor * deferred = nullptr; // abyss: the misses of this layer's FFN that the CPU computes, see moe_defer

        if (n_ffn > 0) {
            cur = build_hc_pre(inpL, attn_mix.pre, il);
            cb(cur, "hc_ffn_pre", il);
        }

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, ffn_mix.post);
        ggml_build_forward_expand(gf, ffn_mix.comb);

        if (n_ffn == 0) {
            cur = ggml_fill(ctx0, attn_out, 0.0f);
            cb(cur, "ffn_out_ced", il);
        } else {
            if (n_ffn < n_tokens) {
                cur = ggml_view_2d(ctx0, cur, n_embd, n_ffn, cur->nb[1], (n_tokens - n_ffn)*cur->nb[1]);
            }

            cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // the reference computes the router in F32: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py#L813
            ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
            ggml_prec_set_acc(logits, GGML_PREC_F32);
            cb(logits, "ffn_moe_logits", il);

            // image (embedding) tokens use the vision router bias, as DeepSeek-V4
            ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;
            if (ubatch.embd != nullptr && layer.ffn_exp_probs_b_vl) {
                exp_probs_b = layer.ffn_exp_probs_b_vl;
            }

            if (defer_ok && il + 1 < n_layer) {
                moe_deferred = &deferred;
            }

            // the reference normalizes only when it routes to more than one expert
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    exp_probs_b,
                    n_expert, hparams.n_expert_used(),
                    LLM_FFN_SILU, hparams.expert_weights_norm && hparams.n_expert_used() > 1,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il, logits);
            moe_deferred = nullptr;
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp, nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);

            if (n_ffn < n_tokens) {
                cur = ggml_pad_ext(ctx0, cur, 0, 0, (int) (n_tokens - n_ffn), 0, 0, 0, 0, 0);
                cb(cur, "ffn_out_ced", il);
            }
        }

        inpL = build_hc_post(cur, residual, ffn_mix.post, ffn_mix.comb, il);
        if (deferred != nullptr) {
            if (moe_defer == 2) {
                inpL = hc_add(deferred, inpL, ffn_mix.post);
            } else {
                defer_sum  = deferred;
                defer_post = ffn_mix.post;
            }
        }
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // no hc_head: collapse with the last FFN's pre-mix
    if (inp_out_ids) {
        inpL     = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens), inp_out_ids), n_embd, hc, inp_out_ids->ne[0]);
        pre_prev = ggml_get_rows(ctx0, pre_prev, inp_out_ids);
    }
    cur = build_hc_pre(inpL, pre_prev, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // abyss: ABYSS_HEAD_SPLIT (llama_model::output_split): a single output's logits are computed on two GPUs at once,
    // the first rows from the copy on the other GPU and the rest on the head's own, and joined on the other GPU (the
    // context's graph callback puts the concat there, so that neither GPU waits for the other before its own rows);
    // each row is the same dot product as before; without the whole head (ABYSS_HEAD_SPLIT_ONLY) every batch goes this
    // way, at most one output per sequence (llama_context limits n_outputs_max)
    if (model.output_split != nullptr && (n_outputs == 1 || model.output == nullptr)) {
        ggml_tensor * rows_split = ggml_mul_mat(ctx0, model.output_split, cur);
        ggml_tensor * rows_rest  = ggml_mul_mat(ctx0, model.output_rest, cur);
        cur = ggml_concat(ctx0, rows_split, rows_rest, 0);
    } else {
        cur = ggml_mul_mat(ctx0, model.output, cur);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
