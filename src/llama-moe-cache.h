#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const;

    // abyss: what the ubatches of up to 8 tokens did so far: selected experts found in the cache, uploaded to it and
    // computed by the CPU
    void small_counts(size_t & hits, size_t & uploaded, size_t & cpu) const;

    // abyss: ABYSS_MOE_BIG_SPLIT=1. A batch too large for the cache uploads the experts it uses from host memory; with
    // the split, the CPU computes the ones with the fewest tokens meanwhile, as a compact batch of (token, expert) pairs.
    // big_split: whether layer il splits a batch of n_tokens
    bool big_split(int32_t il, int64_t n_tokens) const;
    // the plan, computed on the host from the selected experts ids [n_expert_used*n_tokens]: I32 [n + 3*big_cpu_rows()]
    //   [0, n)        the expert the device computes for each pair; for a pair of the CPU, an expert the device uploads
    //                 anyway, whose result big_mask() drops
    //   then m each:  the token of each pair of the CPU, its expert (-1 for padding), and its index among the n pairs
    //                 (n for padding)
    ggml_tensor * build_big_plan(ggml_context * ctx, int32_t il, ggml_tensor * ids) const;
    // F32 [1, n_expert_used, n_tokens]: 1 for the pairs the device computes, 0 for the CPU's
    ggml_tensor * build_big_mask(ggml_context * ctx, ggml_tensor * plan, int64_t n_expert_used, int64_t n_tokens) const;
    int64_t big_cpu_rows() const;
    // for the copy of host weight w that follows a plan of its layer: the experts the device computes
    bool big_used(const ggml_tensor * w, std::vector<bool> & used) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
