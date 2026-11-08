#!/usr/bin/env python3
"""A subset of the MLPerf Agentic Inference dataset for one server and a time budget.

The client (mlcommons/endpoints, AgenticInferenceDataset) sends each client turn (a user or tool row) with the system
prompt and every recorded row before it, so a conversation's longest request is its last client turn. This counts
that request's tokens the way the client's tokenizer does (the model's chat template with the conversation's tools,
thinking on, reasoning effort max), and writes the rows of the chosen conversations in their original order, after the
dataset's metadata row.

    subset.py stats DATASET TOKENIZER_DIR OUT_TSV       per conversation: source, client turns, delays, longest request
    subset.py write DATASET OUT_JSONL ID [ID ...]       the rows of those conversations, contiguous and in order
"""

import json
import sys


def conversations(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    convs = {}
    for r in rows:
        if "conversation_id" in r:  # the first row is the dataset's metadata (the Workato attribution)
            convs.setdefault(r["conversation_id"], []).append(r)
    return rows, convs


def messages_before_last_client_turn(group):
    """The messages of the last client turn's request, built as AgenticInferenceDataset builds them."""
    group = sorted(group, key=lambda r: int(r["turn"]))
    system = next((r["system"] for r in group if r.get("system")), None)
    tools = next((r["tools"] for r in group if r.get("tools")), None)
    history = [{"role": "system", "content": system}] if system else []
    last = None
    for r in group:
        if r.get("tool_results"):
            msgs = [{"role": "tool", "content": t.get("content", ""), "tool_call_id": t.get("tool_call_id")}
                    for t in r["tool_results"]]
        else:
            # a missing field may come as NaN (pandas in the client), which is not equal to itself
            msg = {k: r[k] for k in ("role", "content", "name", "tool_calls", "reasoning_content")
                   if r.get(k) is not None and r.get(k) == r.get(k)}
            if msg.get("role") == "assistant" and "tool_calls" in msg and "content" not in msg:
                msg["content"] = None
            msgs = [msg] if msg.get("role") else []
        if r["role"] in ("user", "tool"):
            last = list(history) + msgs
        history.extend(msgs)
    return last, tools


def stats(path, tok_dir, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    _, convs = conversations(path)
    lines = ["conversation_id\tsource\tclient_turns\tdelay_s\tlongest_request_tokens"]
    for cid, group in convs.items():
        msgs, tools = messages_before_last_client_turn(group)
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                if isinstance(tc.get("function", {}).get("arguments"), str):
                    try:
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                    except ValueError:
                        pass
        ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True,
                                      thinking=True, reasoning_effort="max")
        if hasattr(ids, "keys"):  # transformers 5 returns the encoding, input_ids among its fields
            ids = ids["input_ids"]
        n_client = sum(1 for r in group if r["role"] in ("user", "tool"))
        delay = sum(float(r.get("delay_seconds") or 0) for r in group)
        source = group[0].get("_source", "coding")
        lines.append(f"{cid}\t{source}\t{n_client}\t{delay:.1f}\t{len(ids)}")
        print(lines[-1], flush=True)
    open(out, "w").write("\n".join(lines) + "\n")


def write(path, out, ids):
    rows, convs = conversations(path)
    missing = [i for i in ids if i not in convs]
    assert not missing, f"not in the dataset: {missing}"
    keep = set(ids)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            if r.get("conversation_id") in keep or r.get("_type") == "dataset_metadata":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{out}: {len(ids)} conversations, {sum(len(convs[i]) for i in ids)} rows")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stats":
        stats(*sys.argv[2:5])
    elif cmd == "write":
        write(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        sys.exit(__doc__)
