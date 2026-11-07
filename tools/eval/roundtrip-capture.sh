#!/bin/bash
# Where a conversation's next turn leaves the last answer as the model generated it, in production: after each request
# of the production server ends, while the slot is idle, save slot 0 and keep only its tokens (slot_tokens.py extract),
# the save itself deleted at once, so the end of a capture is an answer as generated and the next capture holds it as
# the client sent it back and the template rendered it. Runs for DURATION seconds (default 3600) or until a capture
# shows a common prefix that ends in the tail of the earlier one MAX_HITS times (default 3); slot_tokens.py compare then
# prints both sides of each divergence. A save takes about 0.13 s, which a request arriving meanwhile waits for.
set -u
E=$(cd "$(dirname "$0")" && pwd)
LOG=$HOME/abyss-stats/server.log
D=${OUT:-$HOME/abyss-eval}/tool-roundtrip-$(date +%m%d-%H%M)
S=$HOME/abyss-stats/slots/cap-roundtrip.bin
mkdir -p "$D"
echo "### capturing into $D from $(date +%T)"
k=0
hits=0
timeout "${DURATION:-3600}" tail -n0 -F "$LOG" 2>/dev/null | grep --line-buffered "stop processing" | while read -r _; do
  curl -s -m 2 http://127.0.0.1:8080/slots | grep -q '"is_processing":false' || continue
  k=$((k + 1))
  curl -s -m 60 -X POST "http://127.0.0.1:8080/slots/0?action=save" -H "Content-Type: application/json" \
    -d '{"filename": "cap-roundtrip.bin"}' > /dev/null
  python3 -I "$E/slot_tokens.py" extract "$S" "$D/$k.tok" > /dev/null && rm -f "$S"
  if [ $k -ge 2 ]; then
    # a common prefix of the last two captures that ends within 300 tokens of the earlier one's end
    if python3 -I - "$D/$((k - 1)).tok" "$D/$k.tok" <<'PY'
import array, sys
a, b = (array.array("i") for _ in range(2))
a.frombytes(open(sys.argv[1], "rb").read()); b.frombytes(open(sys.argv[2], "rb").read())
n = min(len(a), len(b))
lcp = next((i for i in range(n) if a[i] != b[i]), n)
sys.exit(0 if 0 < len(a) - lcp <= 300 else 1)
PY
    then
      hits=$((hits + 1))
      echo "### $(date +%T) capture $k: a common prefix ends in the tail of capture $((k - 1)) ($hits)"
      [ $hits -ge ${MAX_HITS:-3} ] && break
    fi
  fi
done
echo "### done $(date +%T), $(ls "$D"/*.tok 2>/dev/null | wc -l) captures"
python3 -I "$E/slot_tokens.py" compare "$D"
