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

# CPU tables. Rows print as soon as they finish.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset suite --metrics cpu --warmups 0 --repeats 1

# Peak Python memory tables, automatically limited through 70b.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset suite --metrics memory

# Retired-instruction tables.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset suite --metrics instructions

# Kimi-K2: 256 trainer ranks to 128 inference ranks.
.venv/bin/python -m realsim.tools.benchmark_weight_sync_control \
  --preset kimi-k2 --metrics cpu --warmups 0 --repeats 1 --allow-large

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

## Legacy controller, no dedupe

| Model | Trainer publish CPU | `G` lookups CPU | `G` completions CPU | Total CPU | Peak Python memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1b` | 0.738 ms | 9.498 ms | — | 10.254 ms | 116.914 KiB |
| `8b` | 6.332 ms | 87.982 ms | — | 94.334 ms | 300.172 KiB |
| `qwen-27b` | 26.623 ms | 242.357 ms | — | 269.002 ms | 1.004 MiB |
| `70b` | 16.727 ms | 1.009 s | — | 1.026 s | 625.625 KiB |
| `70b-wide` | 129.706 ms | 17.308 s | — | 17.438 s | — |
| `405b` | 134.388 ms | 19.076 s | — | 19.210 s | — |
| `moe` | 278.223 ms | 31.894 s | — | 32.173 s | — |
| `kimi-k2` | 4.105 s | 455.423 s | — | 459.528 s | — |

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
