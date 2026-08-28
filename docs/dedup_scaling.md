# Dedup control-plane benchmark

The benchmark calls the real controller and dedupe control plane directly on one
dedicated OS thread. It does not run the simulator, RPC, transport, or payload copies.

```text
untimed: create fresh controller
        pre-insert persistent trainer key/slice layout
        start controller thread CPU timer
          |
          +-- refresh all T trainer publications
          |
          +-- run all G generator lookups
          |     dedupe: declare pending publication
          |             select sources, route, register readiness gate
          |
          +-- complete all G generators in dependency order
          |     dedupe: publish live metadata
          |             dispatch Published
          |             retire pending metadata and release waiters
          |
        stop controller thread CPU timer
```

Each repeat uses fresh state. The pre-inserted trainer rows model the persistent
layout from the preceding update; generator rows start empty for the new version.
Peak Python memory is the additional traced allocation in a fresh lifecycle after
the trainer pre-insert. CPU, retired instructions, and memory are separate runs, so
`tracemalloc` cannot affect either CPU timing or instruction counting. Memory mode
stops at the `70b` preset because tracing larger lifecycles is disproportionately slow.

## Reusable benchmark

```bash
# Smoke CPU test: all three controller paths.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control --preset smoke

# Historical legacy CPU table. Rows print as soon as they finish.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --historical-torchstore-root ../torchstore \
  --preset suite --variant legacy --metrics cpu --warmups 1 --repeats 3

# Current legacy+dedupe or indexed+dedupe CPU table.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset suite --variant legacy-dedup --metrics cpu --warmups 0 --repeats 1
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset suite --variant indexed-dedup --metrics cpu --warmups 0 --repeats 1

# Peak Python memory tables, automatically limited through 70b.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --historical-torchstore-root ../torchstore \
  --preset suite --variant legacy --metrics memory
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset suite --variant legacy-dedup --metrics memory
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset suite --variant indexed-dedup --metrics memory

# Retired-instruction tables.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset suite --metrics instructions

# Kimi-K2: 256 trainer ranks to 128 inference ranks.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --historical-torchstore-root ../torchstore \
  --preset kimi-k2 --variant legacy --metrics cpu \
  --warmups 0 --repeats 1 --allow-large
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset kimi-k2 --variant legacy-dedup --metrics cpu \
  --warmups 0 --repeats 1 --allow-large
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset kimi-k2 --variant indexed-dedup --metrics cpu \
  --warmups 0 --repeats 1 --allow-large

# One controller path or a custom topology.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --variant indexed-dedup \
  --keys 1199 --source-ranks 8 --generators 8 --generator-shards 4
```

## Benchmark sharding

| Preset | Model class | Keys (`Q`) | Trainer layout | Generator layout |
| --- | --- | ---: | --- | --- |
| `1b` | Llama 1B class | 120 | FSDP 2 = 2 GPUs | TP 2 × DP 4 = 8 GPUs |
| `8b` | Llama 8B class | 290 | FSDP 8 = 8 GPUs | TP 8 × DP 2 = 16 GPUs |
| `qwen-27b` | Qwen3.6-27B | 1,199 | FSDP 4 × TP 2 = 8 GPUs | TP 4 × DP 2 = 8 GPUs |
| `70b` | Llama 70B class | 723 | FSDP 8 = 8 GPUs | TP 8 × DP 8 = 64 GPUs |
| `70b-wide` | Llama 70B class | 723 | FSDP 64 = 64 GPUs | TP 64 × DP 2 = 128 GPUs |
| `405b` | Llama 405B class | 1,500 | FSDP 32 = 32 GPUs | TP 32 × DP 4 = 128 GPUs |
| `moe` | Mixtral / DeepSeek class | 3,000 | FSDP 32 = 32 GPUs | TP 32 × DP 4 = 128 GPUs |
| `kimi-k2` | Kimi-K2 1T | 5,203¹ | FSDP 256 = 256 GPUs | TP 128 × DP 1 = 128 GPUs |

¹ Benchmark key-count assumption; replace it with the deployed state-dict count when available.

## Time complexity

Notation follows the uniform assumption in [toso.md](toso.md), with `Q` used for
`|Q|` and `T = Θ(G)`. Each dedupe lookup includes its pending generator
publication.

| Operation | Legacy, no dedupe | Legacy + dedupe | Indexed + dedupe |
| --- | --- | --- | --- |
| Trainer publish/refresh | `O(GQ)` | `O(GQ)` | `O(GQ)` with an existing geometry index |
| `G` generator lookups | `O(QG²)` | `O(QG² + G² log G)` | `O(GQ log S + G² log G)` |
| `G` generator completions | — | `O(GQ + ΣWₐ)` | `O(GQ + ΣWₐ)` |
| Total | `O(QG²)` | `O(QG² + G² log G + ΣWₐ)` | `O(GQ log S + G² log G + ΣWₐ)` |

## Historical legacy controller, no dedupe

Measured from TorchStore commit `5a4d5d3`.

| Model | Trainer publish CPU | `G` lookups CPU | `G` completions CPU | Total CPU | Peak Python memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1b` | 0.964 ms | 2.907 ms | — | 3.897 ms | 51.453 KiB |
| `8b` | 8.576 ms | 21.438 ms | — | 30.209 ms | 107.516 KiB |
| `qwen-27b` | 35.286 ms | 46.699 ms | — | 81.763 ms | 143.328 KiB |
| `70b` | 20.770 ms | 210.476 ms | — | 231.426 ms | 139.547 KiB |
| `70b-wide` | 168.061 ms | 3.037 s | — | 3.205 s | — |
| `405b` | 176.544 ms | 3.738 s | — | 3.914 s | — |
| `moe` | 361.005 ms | 8.315 s | — | 8.688 s | — |
| `kimi-k2` | 5.923 s | 130.706 s | — | 136.628 s | — |

## Legacy controller + dedupe

| Model | Trainer publish CPU | `G` lookups CPU | `G` completions CPU | Total CPU | Peak Python memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1b` | 0.874 ms | 31.087 ms | 4.661 ms | 36.648 ms | 771.109 KiB |
| `8b` | 6.580 ms | 179.350 ms | 22.996 ms | 208.945 ms | 3.550 MiB |
| `qwen-27b` | 27.911 ms | 467.542 ms | 50.745 ms | 546.221 ms | 8.340 MiB |
| `70b` | 16.680 ms | 3.144 s | 238.667 ms | 3.399 s | 28.408 MiB |
| `70b-wide` | 148.526 ms | 23.287 s | 483.461 ms | 23.919 s | — |
| `405b` | 142.011 ms | 31.536 s | 1.178 s | 32.856 s | — |
| `moe` | 293.646 ms | 63.034 s | 2.295 s | 65.622 s | — |
| `kimi-k2` | 4.448 s | 447.585 s | 4.349 s | 456.382 s | — |

## Indexed controller + dedupe

| Model | Trainer publish CPU | `G` lookups CPU | `G` completions CPU | Total CPU | Peak Python memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1b` | 0.747 ms | 20.254 ms | 17.261 ms | 38.279 ms | 938.070 KiB |
| `8b` | 5.331 ms | 93.667 ms | 86.403 ms | 185.423 ms | 3.486 MiB |
| `qwen-27b` | 26.108 ms | 242.603 ms | 198.261 ms | 467.009 ms | 11.690 MiB |
| `70b` | 12.832 ms | 937.856 ms | 883.553 ms | 1.834 s | 34.015 MiB |
| `70b-wide` | 109.331 ms | 1.900 s | 1.764 s | 3.774 s | — |
| `405b` | 111.710 ms | 3.941 s | 3.881 s | 7.934 s | — |
| `moe` | 229.369 ms | 8.150 s | 8.380 s | 16.759 s | — |
| `kimi-k2` | 3.338 s | 24.223 s | 15.652 s | 43.213 s | — |
