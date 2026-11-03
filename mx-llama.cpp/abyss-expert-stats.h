#pragma once

struct common_params;

// Counts routed-expert picks per layer when ABYSS_EXPERT_STATS names an output JSON file.
// Prefill (ubatch of several tokens) and decode (ubatch of one token) are counted separately.
// The selections come through llama_context_params.cb_moe_ids, after each ubatch.
// ABYSS_EXPERT_STATS_FLUSH_SEC sets how often the file is rewritten (default 30).
void abyss_expert_stats_install(common_params & params);
