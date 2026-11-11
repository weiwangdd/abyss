# 退路脚本

blackwell 上 `~/abyss-stats/restart-*.sh` 的备份（2026-11-11 拷回）。每个脚本是它那项改动上线前的生产启动脚本，留着当退路，之后没有再跟着生产更新，所以用它回退的不只是名字里那一项，而是回到当时的整套配置：

| 脚本 | 写于 | 编译 | 权重 | 上下文 | 专家缓存 | 存取槽位 |
|---|---|---|---|---:|---:|---|
| `restart-nocache.sh` | 10-21 | `build-v41` | 原版 | 128K | 不开 | 否 |
| `restart-noced.sh` | 10-21 | `build-v41` | 原版 | 128K | 14000 | 否 |
| `restart-noprefetch.sh` | 10-23 | `build-ced` | 原版 | 128K | 14000 | 否 |
| `restart-noced-attn.sh` | 10-23 | `build-eng` | 原版 | 128K | 14000 | 否 |
| `restart-noindexer.sh` | 10-24 | `build-b2` | 原版 | 128K | 14000 | 否 |
| `restart-nom2.sh` | 10-25 | `build-m2` | 原版 | 128K | 14000 | 否 |
| `restart-nomerge.sh` | 10-25 | `build-m2` | 原版 | 128K | 14000 | 否 |
| `restart-nochunk.sh` | 10-28 | `build-m2` | 1017 | 192K | 12000 | 是 |
| `restart-cache16.sh` | 10-28 | `build-m2` | 1017 | 192K | 16000 | 是 |
| `restart-nodefer.sh` | 11-03 | `build-m2` | 1017 | 192K | 20500 | 是 |
| `restart-noksplit.sh` | 11-03 | `build-m2` | 1017 | 192K | 20500 | 是 |

10-25 以前的几个用原版权重（mxxm-t 的 GGUF）、不存取槽位；用 `build-m2` 的几个现在跑的是最新编译，只是少开后来的开关。只想关掉现在生产的某一项时，更直接的办法是在 `tools/restart-server.sh` 里去掉那一个开关（各开关的退路见 `mx-llama.cpp/README.md` 和 CLAUDE.md）。
