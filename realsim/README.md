# realsim — a real-code cooperative DES over TorchStore

A single-threaded, deterministic discrete-event simulation that drives the
**real** TorchStore client planning core, the **real** controller directory, and
the **real** in-memory transport/store off-actor, under a virtual clock. It models
only the pieces a capability plugs in: the routing policy and what it executes.

`realsim` is the real-code foundation that [`dedup_sim/`](../dedup_sim/) and
[`kvcache_sim/`](../kvcache_sim/) build on: both `import realsim` and run their
algorithms on the real directory + real types. It deliberately depends on the real
`torchstore` / `torch` / `monarch` install — the client, controller, transport,
and store types that execute are the real ones; only the components being designed
(a routing `Policy`, a capability's `DataPlane`) and the actor/RPC boundary are
substituted with in-process seams.

**See [`../docs/realsim_design.md`](../docs/realsim_design.md) for the full design**
— the concurrency model, how each real object is driven off-actor, the cost model,
the allocation-free data plane, the policy seam, and the concurrency contract.

## What executes

- **Real** `LocalClient` planning core (`_build_volume_requests`,
  `_expand_tensor_slices`, the `asyncio.gather` fan-out, `_assemble_results`).
- **Real** `Controller` directory logic (`_notify_put` / `_notify_delete` over a
  real `Trie`; the two ~5-line read-endpoint bodies are mirrored verbatim).
- **Real** `MonarchRPCTransportBuffer` + `InMemoryStore` put/get lifecycle.
- **Model:** the four types a capability plugs into — `Policy` (which volume
  serves these keys for this requester, and when; consulted *inside* the real
  `locate_volumes`, naive by default), `View` (the read-only observation a policy
  is handed), `DataPlane` (work around and after a transfer) and `Runner`
  (release work items on the virtual clock, install the mesh once, drain).
- **Virtual clock:** every resource cost advances time via `asyncio.sleep` on
  `sim_common.async_engine.AsyncEngine`, so the run is free and deterministic.

## Allocation-free, with fully modeled costs

The sim carries **zero real tensor bytes** and charges **every** resource
analytically from a *target-machine* profile — never measured on the box running
the sim.

- **Data plane.** `--mode meta` (default) uses a `device="meta"` tensor — a real
  `torch.Tensor` with zero storage but exact `shape`/`dtype`/`nbytes`, so a 256 MiB
  *modeled* payload allocates nothing. `--mode metadata` carries no tensor at all:
  a `(shape, dtype)` `TensorDescriptor` stands in for the payload and round-trips
  the real object put/get path.
- **Cost model (`sim_common/cost_model.py`).** A `MachineProfile` supplies all
  target-hardware constants (per-tier network `(latency, bandwidth)`, RAM, storage,
  GPU/CPU flops + memory bandwidth). Analytic functions — `network_time`,
  `mem_copy_time`, `storage_time`, and a roofline `compute_time` — turn modeled
  `nbytes`/`flops` into virtual time. `DEFAULT_PROFILE` is an illustrative demo
  profile, not measured.
- **Full resource exercise.** One burst charges **compute/GPU** (the producer's
  generate step) + **network** (client↔volume fabric) + **storage** (write on put,
  read on serve) + **RAM** (host staging on serve), all through the cost model off
  one `MachineProfile`.

## Environment

Needs the venv that has torchstore/torch/monarch built (see the repo root
[`README.md`](../README.md) "Building the live example from source"). `realsim` is
**not** stdlib-only. Run from the repo directory with that interpreter on
`PYTHONPATH`.

## Running the demo

```
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.run_realsim
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.run_realsim -m 4 -v
```

- `-m/--readers N` -- readers in the burst (default 3).
- `-n/--elements N` -- elements in `W` (float32; payload = `4*N` bytes).
- `--mode meta|metadata` -- the allocation-free data-plane carrier: `meta`
  (zero-storage meta tensor, default) or `metadata` (a `(shape, dtype)`
  descriptor, no tensor at all).
- `--seed S` -- switch the engine to seeded-random ready-queue mode (default:
  FIFO, reproducible).
- `-v` -- also print the full per-event virtual-time trace (DEBUG).

Output: the fabric/wallclock summary + an ASCII source→dest tree at INFO. Under the
naive policy every reader pulls the origin (`m×` fabric) -- the baseline a
read-through policy would cut toward the 1× union.

## Testing

```
# whole cross-package suite (realsim + shared engine + both capability sims)
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m pytest \
  realsim/tests sim_common/tests dedup_sim/tests kvcache_sim/tests -q

# just realsim
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m pytest realsim/tests -q
```

Tests are deterministic (byte-identical traces across runs; invariants across a
couple of seeds under random scheduling):

- **`test_correctness.py`** — off-sim byte-level reassembly on tiny **real** CPU
  tensors put/got through the *same* real client/controller/`InMemoryStore` code.
- **`test_perf.py`** — the perf guard: a 256 MiB *modeled* payload must not move
  peak RSS, and a realsim run must stay within a tolerant multiple of a `dedup_sim`
  run's wall + RSS (measured in fresh subprocesses).
- **`test_composability.py`** — imports realsim's real-directory backend
  (`RealControllerAdapter` / `FakeControllerHandle`) standalone and exercises it.

## Concurrency-contract lint

`realsim/tools/check_contract.py` fails the build if any simulated path in the
scanned packages reaches for a determinism-breaking primitive (threads, forks,
`time.sleep`, wall-clock reads in library code, unseeded randomness), **or** if a
capability's `control/` module imports the executing half (a `data/` package, the
mesh, or a store client). `asyncio.sleep` (virtual clock) and seeded
`random.Random(seed)` are allowed. It is wired into `tests/test_contract.py` and
also runs standalone:

```
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.tools.check_contract
```

## Layout

```
realsim/
  seams/          in-process adapters onto real torchstore surfaces;
                  transport charges the cost model per put/get; factory.py is the
                  one place create_transport_buffer is substituted
  adapters/       thin wiring that constructs the real objects off-actor
  mesh.py         Mesh -- the multi-client wiring a capability builds on: per-node
                  volumes + real clients, one directory, one resource registry,
                  one shared transport factory. It is also the Deployment a
                  capability's data plane runs against (client_for resolves a node)
  runner.py       Runner -- release work items on the virtual clock in
                  (release_time, id) order, install the mesh once, gather, drain
  scenarios/      put_get.py: seed a key, then m clients get it; meta/metadata data
                  plane + full resource-cost exercise
  run_realsim.py  the demo entrypoint
  tools/          check_contract.py: the concurrency + plane-separation lint
  tests/          seams smoke, determinism, contract lint, off-sim correctness,
                  perf guard, composability, mesh wiring, the shared plane types
proposed/       every contract that outlives the simulator; imports nothing
  policy.py       Policy.select(view, keys, requester) -> ranked sources +
                  readiness, plus notice() to open a readiness gate. Naive (all
                  holders, directory order) is the default; the controller
                  consults it inside locate_volumes
  view.py         View -- awaited, read-only observation: locate, topology and
                  locality, the clock. Built over a Directory protocol
  deployment.py   Deployment -- how data-plane code reaches its store
  plane.py        DataPlane -- execute(item) / after(item, result), both
                  defaulting to no-op
  cost.py         TransferCost -- what a fetch is predicted to cost
  topology.py     Endpoint / Tier / locality -- where a volume is
domain/
  llm.py          Model -- a transformer reduced to what a sim charges against
                  (flops/token, KV bytes/token) -- plus prefill/decode-step times.
                  Domain facts: not sim machinery, not policy
sim_common/
  async_engine.py deterministic asyncio loop + virtual clock
  cost_model.py   MachineProfile + analytic network/RAM/storage/CPU/GPU costs
  report.py       Ledger (transfer edges + byte counters + outcome rows +
                  aggregations) and the source->dest tree renderer
  engine.py trace.py topology.py   shared DES library (reused)
```
