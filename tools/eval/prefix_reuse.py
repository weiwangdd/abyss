#!/usr/bin/env python3
"""How much of each request's prompt the server took from its cache, against what it could have, from server logs
(the default verbosity is enough): per request the slot selection (by LCP, with f_sim = the common prefix over the new
prompt, or by LRU), the prompt length P (the context at release minus the tokens generated), the tokens computed C,
and so the tokens reused R = P - C. A request selected by LCP shares f_sim * P tokens with the slot; a DSV4 slot goes
back to its newest checkpoint before that. Requests that left more than --waste tokens of the common prefix unused, and
the requests computed whole (LRU, or LCP with nothing reused) of more than --whole tokens, are listed; the log's start
comes from its modification time and its last relative time stamp.

Follow-up turns are the requests selected by LCP that keep at least half of the slot, which still holds the previous
request of the same log: for them the tokens the slot held that the turn computed again (before mx-llama.cpp/0041
mostly the last answer and 4 tokens, which the next turn always computed again), against all they computed. Logs of
0041 have a line per request with the exact common prefix X, the slot's tokens Y and where it continued Z, which sorts
the follow-up turns by where they continued: from the slot's end (the prompt extends the whole slot), from a checkpoint
in the last answer (ABYSS_GEN_CKPT, when the prompt leaves the answer before its end), or from the end of the last
prompt or before (as before 0041; counted apart are the turns whose common prefix ends before the last answer, an
earlier part of the conversation sent another way, and those whose common prefix ends in the answer's first 16
tokens, such as reasoning the template drops). The checkpoints made in answers are summed with their time. --since and
--until keep the requests of a time span ("YYYY-MM-DD HH:MM"), for example before and after a change.

    prefix_reuse.py [--waste 2000] [--whole 10000] [--since T] [--until T] <server log>...
"""

import argparse
import datetime
import os
import re
import statistics


def stamp(s):
    mins, secs, ms, us = s.split(".")
    return datetime.timedelta(minutes=int(mins), seconds=int(secs), milliseconds=int(ms), microseconds=int(us))


def parse(path):
    reqs, cur, last = [], {}, None
    for line in open(path, errors="replace"):
        m = re.match(r"(\d+\.\d+\.\d+\.\d+) ", line)
        if m:
            last = stamp(m.group(1))
        if "selected slot by LCP similarity" in line:
            m = re.search(r"f_sim_best = ([\d.]+).*f_keep = ([\d.]+)", line)
            cur = {"sel": "LCP", "fsim": float(m.group(1)), "fkeep": float(m.group(2)), "t": last}
        elif "selected slot by LRU" in line:
            cur = {"sel": "LRU", "t": last}
        elif "saving it in the prompt cache" in line:
            m = re.search(r"keeps (\d+) tokens of a prompt that reached (\d+)", line)
            cur["kept"], cur["slot"] = int(m.group(1)), int(m.group(2))
        elif "swapping it in" in line:
            cur["swapped"] = True
        elif "abyss: the prompt of" in line:
            m = re.search(r"the prompt of (\d+) tokens shares (\d+) with the slot's (\d+), continuing from (\d+)", line)
            cur.update(N=int(m.group(1)), X=int(m.group(2)), Y=int(m.group(3)), Z=int(m.group(4)))
        elif "checkpoints in the answer" in line:
            m = re.search(r"abyss: (\d+) checkpoints in the answer, ([\d.]+) ms", line)
            cur["ckpts"], cur["ckpt_ms"] = int(m.group(1)), float(m.group(2))
        elif "prompt eval time =" in line:
            m = re.search(r"= +([\d.]+) ms / +(\d+) tokens", line)
            cur["ms"], cur["C"] = float(m.group(1)), int(m.group(2))
        elif "       eval time =" in line:
            m = re.search(r"= +([\d.]+) ms / +(\d+) tokens", line)
            cur["G"] = int(m.group(2))
        elif "stop processing: n_tokens" in line:
            m = re.search(r"n_tokens = (\d+)", line)
            if "C" in cur and "G" in cur and "t" in cur:
                cur["T"] = int(m.group(1))
                cur["P"] = cur["T"] - cur["G"]
                cur["R"] = cur["P"] - cur["C"]
                reqs.append(cur)
            cur = {}
    start = datetime.datetime.fromtimestamp(os.path.getmtime(path)) - (last or datetime.timedelta(0))
    for r in reqs:
        r["when"] = start + r["t"]
    return reqs


def follow_ups(reqs):
    """The follow-up turns as (request, previous request), each pair from one log."""
    out = []
    for prev, r in zip(reqs, reqs[1:]):
        if prev["log"] != r["log"] or r["sel"] != "LCP" or r["fkeep"] < 0.5 or r.get("swapped"):
            continue
        if "Y" in r and r["Y"] != prev["T"]:
            continue  # the slot holds something else than the previous request
        out.append((r, prev))
    return out


def report_follow_ups(pairs, reqs):
    if not pairs:
        print("\nno follow-up turns")
        return
    old = lambda r, prev: max(0, prev["T"] - (r["Z"] if "Z" in r else r["R"]))
    n_c = sum(r["C"] for r, _ in pairs)
    n_old = sum(old(r, p) for r, p in pairs)
    ms = [r["ms"] for r, _ in pairs]
    print(f"\nfollow-up turns: {len(pairs)}, prefill {sum(ms) / 1e3:.0f} s (per turn mean {statistics.mean(ms) / 1e3:.2f} s, "
          f"median {statistics.median(ms) / 1e3:.2f} s), {n_c} tokens computed, of them {n_old} ({n_old / max(1, n_c):.0%}) "
          f"the slot held")
    exact = [(r, p) for r, p in pairs if "Z" in r]
    if not exact:
        return
    # the previous prompt's length: its own line if it has one, else its context at release minus its answer
    p_len = lambda p: p["N"] if "N" in p else p["P"] + 1
    kinds = {"from the slot's end": [], "from a checkpoint in the last answer": [],
             "from the end of the last prompt or before": []}
    for r, p in exact:
        if r["Z"] == r["Y"]:
            kinds["from the slot's end"].append((r, p))
        elif r["Z"] > p_len(p):
            kinds["from a checkpoint in the last answer"].append((r, p))
        else:
            kinds["from the end of the last prompt or before"].append((r, p))
    print(f"  where the {len(exact)} with the line of mx-llama.cpp/0041 continued:")
    for k, v in kinds.items():
        if not v:
            print(f"    {k:42s}    0 turns")
            continue
        again = [r["X"] - r["Z"] for r, _ in v]  # tokens of the common prefix computed again
        note = ""
        if k.startswith("from the end of the last prompt"):
            before = sum(1 for r, p in v if r["X"] < p_len(p))
            start = sum(1 for r, p in v if 0 <= r["X"] - p_len(p) <= 16)
            note = (f"; the common prefix ends before the last answer in {before}, "
                    f"in the answer's first 16 tokens in {start}")
        print(f"    {k:42s} {len(v):4d} turns, prefill mean {statistics.mean(r['ms'] for r, _ in v) / 1e3:5.2f} s, "
              f"{sum(r['C'] for r, _ in v):7d} tokens computed, {sum(again):7d} of the common prefix again "
              f"(median {statistics.median(again):.0f}){note}")
    ck = [r for r in reqs if "ckpts" in r]
    n_ck = sum(r["ckpts"] for r in ck)
    if n_ck:
        print(f"  checkpoints made in answers (ABYSS_GEN_CKPT): {n_ck} in {len(ck)} requests, "
              f"{sum(r['ckpt_ms'] for r in ck):.0f} ms ({sum(r['ckpt_ms'] for r in ck) / n_ck:.2f} ms each)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--waste", type=int, default=2000)
    ap.add_argument("--whole", type=int, default=10000)
    ap.add_argument("--since", type=datetime.datetime.fromisoformat)
    ap.add_argument("--until", type=datetime.datetime.fromisoformat)
    a = ap.parse_args()
    reqs = []
    for p in a.logs:
        reqs += [dict(r, log=os.path.basename(p)) for r in parse(p)]
    reqs.sort(key=lambda r: r["when"])
    # the follow-up pairs need the request before the span's first one
    pairs = [(r, p) for r, p in follow_ups(reqs)
             if (not a.since or r["when"] >= a.since) and (not a.until or r["when"] < a.until)]
    reqs = [r for r in reqs if (not a.since or r["when"] >= a.since) and (not a.until or r["when"] < a.until)]
    if not reqs:
        print("no requests")
        return
    total_ms = sum(r["ms"] for r in reqs)
    print(f"{len(reqs)} requests from {reqs[0]['when']:%m-%d %H:%M} to {reqs[-1]['when']:%m-%d %H:%M}, "
          f"prefill {total_ms / 1e3:.0f} s in all, {sum(r['C'] for r in reqs)} tokens computed of "
          f"{sum(r['P'] for r in reqs)} ({sum(r['R'] for r in reqs) / max(1, sum(r['P'] for r in reqs)):.1%} reused)")
    waste, whole = [], []
    for r in reqs:
        if r["sel"] == "LCP":
            # exact with the line of 0041, else from f_sim
            lcp = r["X"] if "X" in r else r["fsim"] * r["P"]
            r["waste"] = lcp - (r["Z"] if "Z" in r else r["R"])
            if r["waste"] > a.waste:
                waste.append(r)
        if r["C"] >= r["P"] - 8 and r["C"] > a.whole:
            whole.append(r)
    print(f"\nLCP selections that left more than {a.waste} tokens of the common prefix unused: {len(waste)}, "
          f"{sum(r['ms'] for r in waste) / 1e3:.0f} s of prefill")
    for r in waste:
        lcp = r["X"] if "X" in r else r["fsim"] * r["P"]
        print(f"  {r['when']:%m-%d %H:%M:%S} {r['log']:16s} prompt {r['P']:7d}, common prefix ~{lcp:7.0f} "
              f"(f_keep {r['fkeep']:.3f}{', slot ' + str(r['slot']) + ' kept ' + str(r['kept']) if 'kept' in r else ''}), "
              f"reused {r['R']:7d}, computed {r['C']:7d} in {r['ms'] / 1e3:6.1f} s")
    print(f"\nprompts of more than {a.whole} tokens computed whole: {len(whole)}, {sum(r['ms'] for r in whole) / 1e3:.0f} s")
    for r in whole:
        print(f"  {r['when']:%m-%d %H:%M:%S} {r['log']:16s} {r['sel']}{' f_sim %.3f' % r['fsim'] if r['sel'] == 'LCP' else ''}"
              f" prompt {r['P']:7d} in {r['ms'] / 1e3:6.1f} s")
    report_follow_ups(pairs, reqs)


if __name__ == "__main__":
    main()
