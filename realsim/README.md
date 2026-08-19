# realsim — a real-code cooperative DES over TorchStore

A single-threaded, deterministic discrete-event simulation that drives the
**real** TorchStore client planning core, the **real** controller directory, and
the **real** in-memory transport/store off-actor, under a virtual clock. It models
only the pieces a capability plugs in: the control planes it asks and what it
executes.

`realsim` is the real-code foundation that [`putget_sim/`](../putget_sim/),
[`dedup_sim/`](../dedup_sim/) and [`kvcache_sim/`](../kvcache_sim/) build on: all
three `import realsim` and run their algorithms on the real directory + real
types. It is the foundation only — it owns no scenario and no demo; the unrouted
put/get burst is `putget_sim`. It deliberately depends on the real
`torchstore` / `torch` / `monarch` install — the client, controller, transport,
and store types that execute are the real ones; only the components being designed
(a routing `KeySelector`, a capability's `DataPlane`) and the actor/RPC boundary are
substituted with in-process seams.

**See [`../docs/des_design.md`](../docs/des_design.md) for the full design**
— the concurrency model, how each real object is driven off-actor, the cost model,
the allocation-free data plane, the control-plane seams, and the concurrency contract.

## What executes

- **Real** `LocalClient` planning core (`_build_volume_requests`,
  `_expand_tensor_slices`, the `asyncio.gather` fan-out, `_assemble_results`).
- **Real** `Controller` directory logic (`_notify_put` / `_notify_delete` over a
  real `Trie`; the two ~5-line read-endpoint bodies are mirrored verbatim).
- **Real** `MonarchRPCTransportBuffer` + `InMemoryStore` put/get lifecycle.
- **Model:** the four types a capability plugs into — `KeySelector` (which volume
  serves these keys for this requester, and when; a service the data plane asks
  before it reads, naive by default), `Environment` plus typed sensors (stable run
  facts beside evolving observations), `DataPlane` (a capability's executing half) and
  `Runner` + `ItemDispatch` (release work items on the virtual clock, install the
  mesh once, gather).
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

## Running a demo

`realsim` has no `__main__` of its own — a run is always some sim's. The one that
exercises this package and nothing else is
[`putget_sim`](../putget_sim/README.md):

```
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m putget_sim
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m putget_sim -m 4 -v
```

## Testing

```
# whole cross-package suite (realsim + shared engine + the capability sims)
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m pytest \
  realsim/tests sim_common/tests dedup_sim/tests kvcache_sim/tests -q

# just realsim
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m pytest realsim/tests -q
```

Tests are deterministic (byte-identical traces across runs; invariants across a
couple of seeds under random scheduling). They drive `putget_sim`'s
capability-free fixture, so they exercise the whole stack without depending on a
capability's decisions:

- **`test_correctness.py`** — off-sim byte-level reassembly on tiny **real** CPU
  tensors put/got through the *same* real client/controller/`InMemoryStore` code.
- **`test_perf.py`** — the perf guard: a 256 MiB *modeled* payload must not move
  peak RSS, and a `putget_sim` run must stay within a tolerant multiple of a
  `dedup_sim` run's wall + RSS (measured in fresh subprocesses).
- **`test_composability.py`** — imports realsim's real-directory backend
  (`RealControllerAdapter` / `LocalControllerHandle`) standalone and exercises it.
- **`test_demos.py`** — every sim's `Demo` declares its parts (the ABC refuses
  otherwise) and every scenario of every sim runs end to end.

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

## Structure lint

`realsim/tools/check_structure.py` fails the build if a sim package drifts out of
shape: a missing part (`__main__.py`, `README.md`, `workload/`, `report/`), a
`control/` without a `data/`, a folder-private module whose name does not say so
(`_thing.py`), a public function no other module uses (same remedy: `_thing`), a
README layout block that names a file which does not exist — or omits one that
does — or a module whose `__all__` is absent, names something that
is not there, or omits something public. A test importing a name does not make it
surface: that is how a helper with no callers stays alive, and it is what the
name rule is for. `PUBLIC_ANYWAY` and `PUBLIC_NAMES` are the explicit exception
lists for the two privacy rules; `__init__.py` and `__main__.py` are exempt
from the `__all__` rule, since a package's `__all__` is a curated re-export list
rather than a mirror of its own contents.

The `Demo` contract is deliberately *not* linted: `realsim.demo.Demo` is an ABC,
so a demo that has not declared its scenarios cannot be constructed, and
`tests/test_demos.py` constructs and runs all three.

```
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.tools.check_structure
```

## Text-diagram renderer

`realsim/tools/text_diagram.py` parses a strict XML-like DSL. Its supported elements
are `diagram`, `stack`, `row`, `box`, `text`, `place-line`, `place-lines`, `line`,
`between`, and `at`; unknown HTML features fail. The source declares content and
layout, and the renderer replaces the marked sections in the Markdown document:

```
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.tools.text_diagram \
  docs/sensor_environment_selector_flow.diagram.xml
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
                  (release_time, id) order, install the mesh once, gather
  simulation.py   Simulation -- assembles engine + mesh + directory + registry,
                  and runs a Workload's items on it
  run.py          the run lifecycle in one place -- Workload (the work a run
                  performs), Run (one labelled configuration, which knows how to
                  execute() itself), Result, Report. The only way anything runs,
                  so no capability wires its own stack
  demo.py         Demo / Scenario / Console -- a sim's command line, declared,
                  plus the run flags/logging every one of them shares. A Scenario
                  declares its Runs and narrates the Results; Demo.main is the
                  one place that executes
  tools/          check_contract.py: the concurrency + plane-separation lint
                  check_structure.py: the shape of a sim package
                  text_diagram.py: strict XML-like monospaced-diagram renderer
                  parity.py: every run of every scenario over a knob matrix, one
                  diffable line each (fingerprint + headline metrics) -- what says
                  whether a change moved a number, by diffing two checkouts' output
  tests/          seams smoke, determinism, contract lint, off-sim correctness,
                  perf guard, composability, mesh wiring, the shared plane types
putget_sim/     the unrouted put/get burst (no control plane, no data plane) -- the m x
                baseline, and the fixture realsim's own tests drive
  workload/       put_get.py: seed a key, then m clients get it; meta/metadata
                  carrier + full resource-cost exercise. scenarios.py: its Runs
  report/         summary.py: fabric/wallclock summary + source->dest tree
  __main__.py     the Demo declaration (`python -m putget_sim`)
proposed/       every contract that outlives the simulator; imports nothing
  selector.py     KeySelector.select(keys, requester) -> ranked sources, once
                  they are usable. Naive is all holders in directory order. The
                  data plane asks it and passes the answer to an ordinary read
                  (prefer(): what the store does with a preference)
  environment.py  Environment -- topology, profile, transfer pricing and clock
  sensors.py      DirectorySensor -- pinned directory reads, one-pass TorchStore
                  coverage and ordered fetch planning, with unchanged live coverage
                  reused; LoadSensor -- the common load reading used by Balance
  deployment.py   Deployment -- how data-plane code reaches its store, the one
                  control plane it asks (control_plane_handle, whatever that plane
                  declares) and another node's data plane (plane_handle, for whoever
                  follows an address one of them answered with); and each service as
                  a caller reaches it -- Controller,
                  StorageVolume and Sensor (facts a decision reads, the load a
                  store cannot see, reached only in the process that holds it)
  dispatch.py     Dispatcher -- where a host's facts arrive, and one action, every
                  reducer that folds its type, one commit, and one payload-free
                  gate update at it. Holds no application state itself
  plane.py        ControlPlane -- attach(environment, sensors) + dispatcher, the one a run puts a
                  service in front of; DataPlane -- attach(deployment), routes,
                  and no verbs: what a capability does is its own to name
  routed.py       routed() -- a data plane declaring that a member may answer with
                  the ADDRESS of the host a request belongs on, and where in that
                  answer it is; RoutedPlane -- a caller that calls the member again
                  there, over plane_handle, so nobody writes the following and no
                  host holds a peer (peerless)
  environment.py  topology + MachineProfile-backed read pricing + clock
  sensors.py      pinned directory reads and the common load observation
  topology.py     Endpoint / Tier / locality -- where a volume is
domain/
  llm.py          Model -- a transformer reduced to what a sim charges against
                  (flops/token, KV bytes/token) -- plus prefill/decode-step times.
                  Domain facts: not sim machinery, not selector
sim_common/
  async_engine.py deterministic asyncio loop + virtual clock
  cost_model.py   MachineProfile + analytic network/RAM/storage/CPU/GPU costs
  report.py       Ledger (transfer edges + byte counters + outcome rows +
                  aggregations) and the source->dest tree renderer
  trace.py topology.py   shared DES library (reused)
  engine.py              the ancestor callback DES (Sim/Promise): reference only,
                         not on the sim path and imported by nothing
```
