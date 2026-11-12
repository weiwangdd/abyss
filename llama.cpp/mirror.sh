#!/bin/bash
# Mirrors the port branch into a history that fits this repository: a root commit with the tree of the upstream commit
# the port is based on, without upstream's history, and one commit per commit of the port on top of it, each with the
# same tree, message and dates, authored as AUTHOR. Run in the llama.cpp tree with the port (blackwell:
# ~/src/llama.cpp-abyss); it sets the branch MIRROR (default llama-cpp-mirror) there and writes it to a bundle:
#
#     llama.cpp/mirror.sh [bundle, default /tmp/llama-cpp-mirror.bundle]
#     # in this repository, with the bundle copied over:
#     git fetch <bundle> llama-cpp-mirror:llama-cpp
#
# Everything a commit is made of is taken from the port, so a second run gives the same commits, and after new commits
# on the port the fetch is a fast-forward. A rebase of the port onto another upstream commit gives a new history.
set -eu
BASE=${BASE:-2cbe0ec}
PORT=${PORT:-abyss-port}
MIRROR=${MIRROR:-llama-cpp-mirror}
AUTHOR_NAME=${AUTHOR_NAME:-Wei Wang}
AUTHOR_EMAIL=${AUTHOR_EMAIL:-72610094+weiwangdd@users.noreply.github.com}
bundle=${1:-/tmp/llama-cpp-mirror.bundle}

export GIT_AUTHOR_NAME=$AUTHOR_NAME GIT_AUTHOR_EMAIL=$AUTHOR_EMAIL
export GIT_COMMITTER_NAME=$AUTHOR_NAME GIT_COMMITTER_EMAIL=$AUTHOR_EMAIL

base=$(git rev-parse "$BASE^{commit}")
date=$(git log -1 --format=%cI "$base")
parent=$(printf '%s\n\n%s\n\n%s\n' \
  "llama.cpp : import the upstream tree at ${base:0:7}" \
  "The tree of https://github.com/ggml-org/llama.cpp at commit $base ($(git log -1 --format=%cs "$base")), without its history. The commits on top of it are the Abyss patches, one commit each." \
  "Assisted-by: Claude" |
  GIT_AUTHOR_DATE=$date GIT_COMMITTER_DATE=$date git commit-tree "$base^{tree}" -F -)

n=0
for c in $(git rev-list --reverse "$base..$PORT"); do
  # the message as it is stored, after the header of the commit object
  parent=$(git cat-file commit "$c" | sed '1,/^$/d' |
    GIT_AUTHOR_DATE=$(git log -1 --format=%aI "$c") GIT_COMMITTER_DATE=$(git log -1 --format=%cI "$c") \
    git commit-tree "$c^{tree}" -p "$parent" -F -)
  n=$((n + 1))
done
[ "$(git rev-parse "$parent^{tree}")" = "$(git rev-parse "$PORT^{tree}")" ]
git branch -f "$MIRROR" "$parent"
git bundle create "$bundle" "$MIRROR" 2> /dev/null
echo "$MIRROR: $n commits on the tree of ${base:0:7}, tip $(git rev-parse --short "$parent"), bundle $bundle ($(du -h "$bundle" | cut -f1))"
