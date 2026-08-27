# Precomputed routing simulation

This scenario imports `RoutingPlan`, `RoutingService`, and `RoutingClient` from
`torchstore.routing`. Toso contains only the workload, simulator adapters, and
benchmark assertions; the routing implementation lives in TorchStore.

The ordinary one-key dedupe scenario also runs through precomputed routing:

| Path | Origin bytes | Total delivered | Completion |
| --- | ---: | ---: | ---: |
| Naive | 192 B | 192 B | 0.0195 s |
| Online dedupe, fan-out 2 | 64 B | 192 B | 0.0401 s |
| Precomputed routing | 64 B | 192 B | 0.0327 s |

Both dedupe implementations reduce origin traffic from 3× to 1×. The online
path chooses sources from live directory state; precomputed routing installs the
same one-ingress tree before the burst and performs only local lookups during it.

The integrated scenario reuses the existing `WeightSync` workload, `Run`,
`ItemDispatch`, real TorchStore volumes and transport lifecycle, virtual clock,
machine profile, ledger, and shared-egress resource model:

```text
PYTHONPATH=. .venv/bin/python -m dedup_sim routing
```

For Qwen3.6-27B (55.6 GB), TP=4, DP=2, effective trainer-to-generator
bandwidth of 17.5 GB/s, and generator relay bandwidth of 900 GB/s:

| Path | Trainer bytes | Generator relay bytes | Completion |
| --- | ---: | ---: | ---: |
| Every generator reads trainers | 111.2 GB | 0 GB | 6.354 s |
| Precomputed routing | 55.6 GB | 55.6 GB | 3.193 s |

```text
55.6 GB / 17.5 GB/s + (55.6 GB / 4) / 900 GB/s = 3.193 s
```

The adapter constructs the existing TorchStore transport directly for the
source volume named by the local route. Trainer publication and generator
read-through use direct volume puts, and receives use direct volume gets. The
controller directory remains empty.

The scenario represents the trainer side as one aggregate constrained egress
endpoint. If trainer ranks have independent links, their assigned segments can
progress in parallel.

Run the focused test with:

```text
PYTHONPATH=. .venv/bin/python -m pytest dedup_sim/routing/tests -q
```
