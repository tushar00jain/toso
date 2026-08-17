# Deterministic simulation design

<!-- Generated from des_design.diagram.xml by realsim.tools.text_diagram. -->

`realsim` runs real TorchStore planning and storage code under a deterministic
virtual clock. Capability packages provide only the decision and execution logic
being studied; `sim_common` supplies time, costs, resources, tracing, and reports.

## 1. What belongs where

<!-- text-diagram:stack:start -->
```
┌───────── CAPABILITY ──────────┐   ┌───────────── REALSIM ─────────────┐   ┌────────── REAL TORCHSTORE ───────────┐
│ Workload / Scenario           │──►│ Runner / Simulation               │──►│ Request / tensor metadata            │
│ ControlPlane + selectors      │──►│ control + dispatcher services     │──►│ Controller directory                 │
│ DataPlane                     │──►│ Deployment / Mesh                 │──►│ LocalClient planning                 │
│                               │   │ volume + transport seams          │──►│ StorageVolume / InMemoryStore        │
│ Report                        │──►│ Ledger / Trace                    │   │                                      │
│ only capability-specific code │   │ assembly, seams, real adapters    │   │ production algorithms and types      │
└───────────────────────────────┘   └───────────────────────────────────┘   └──────────────────────────────────────┘
┌──────────────────────────────────────────────── VIRTUAL MACHINE ─────────────────────────────────────────────────┐
│ AsyncEngine clock  +  ResourceRegistry  +  MachineProfile costs  +  deterministic Trace                          │
│ sim_common owns time and accounting; it knows no capability vocabulary                                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
<!-- text-diagram:stack:end -->

The production-shaped path is real: `LocalClient` planning, the `Controller` trie,
`Request`/`StorageInfo`/`TensorSlice`, `StorageVolume`, transport-buffer lifecycle,
and `InMemoryStore`. In-process handles replace actor RPC, and the capability's
control/data planes are the application code under study.

The layering rule is simple: `proposed` imports nothing; `sim_common` knows only
simulation primitives; `realsim` assembles the real stack; capability packages
depend on that foundation. Control decides through a read-only `View`; data moves
bytes through a `Deployment`.

## 2. One work item

<!-- text-diagram:work-item:start -->
```
┌── REQUEST ───┐   ┌────── CONTROL ──────┐   ┌────── DATA ──────┐   ┌────── TORCHSTORE ───────┐   ┌──── VIRTUAL TIME ─────┐
│ WorkItem     │   │ View → Selector     │   │ act on answer    │   │ LocalClient → Controller│   │ ResourceRegistry claim│
│ released at t│──►│ Selection / Response│──►│ ordinary APIs    │──►│ → Volume / Transport    │──►│ asyncio.sleep(cost)   │
│              │   │ commit Action       │   │ report facts     │   │ move + register bytes   │   │ Ledger / Trace        │
│ Runner       │   │                     │   │                  │   │                         │   │                       │
└──────────────┘   └─────────────────────┘   └──────────────────┘   └─────────────────────────┘   └───────────────────────┘
┌────────── DIRECTORY TRUTH ──────────┐   ┌────────── SENSOR TRUTH ───────────┐   ┌────────── NEXT DECISION ───────────┐
│ put / delete → key → current holders│   │ Action → Dispatcher → sensor folds│   │ View reads directory + sensor state│
└─────────────────────────────────────┘   └───────────────────────────────────┘   └────────────────────────────────────┘
```
<!-- text-diagram:work-item:end -->

The directory and sensors are separate truths. Store puts and evictions change
residency. Dispatched actions change a capability's queue, reservation, routing,
or load model. The next decision reads both through its view; selectors mutate
neither.

## 3. Time and cost

<!-- text-diagram:cost:start -->
```
┌─────── WORK ────────┐   ┌───── PROFILE ──────┐   ┌────── COST ───────┐   ┌──── RESOURCE ─────┐   ┌─────── CLOCK ───────┐
│ nbytes / flops      │   │ latency / bandwidth│   │ network / copy    │   │ claim contention  │   │ asyncio.sleep       │
│ source + destination│──►│ RAM / storage / GPU│──►│ storage / roofline│──►│ ordered completion│──►│ advance virtual time│
└─────────────────────┘   └────────────────────┘   └───────────────────┘   └───────────────────┘   └─────────────────────┘
┌──────────────────────────────────────────────────── CARRIER ─────────────────────────────────────────────────────┐
│ meta tensor or TensorDescriptor: exact shape / dtype / modeled bytes, zero payload storage                       │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
<!-- text-diagram:cost:end -->

All modeled delay is `asyncio.sleep` on `AsyncEngine`. Same-time callbacks and
timers are ordered by `(time, seq)`; an optional seed shuffles the ready queue for
repeatable interleaving exploration. The same `MachineProfile` feeds control-plane
prediction and data-plane charges.

Payloads allocate no real storage. Meta tensors preserve real tensor type,
shape, dtype, and byte count; `TensorDescriptor` exercises the object path with
shape and dtype only. Both drive the real client/controller/store path.

The concurrency contract makes a single-thread cooperative DES sound for this
stack: mutable state is actor-owned, interaction crosses awaited seams, and task
switches occur only at suspension points. The contract lint rejects threads,
forks, blocking sleeps, wall-clock control flow, unseeded randomness, and
control-plane imports of execution machinery.

## 4. The three simulations

<!-- text-diagram:capabilities:start -->
```
┌────────── PUTGET ───────────┐   ┌─────────── DEDUP ───────────┐   ┌────────── KVCACHE ───────────┐
│ no ControlPlane             │   │ sources(keys, requester)    │   │ decide(request) + sources    │
│ no DataPlane                │   │ FanoutSensor + readiness    │   │ queue / decode / pull sensors│
│ every reader pulls origin   │   │ preferred get → local put   │   │ fetch → prefill → decode     │
│                             │   │                             │   │                              │
│ baseline: m× fabric         │   │ goal: 1× origin fabric      │   │ TTFT / TBT / hit rate / SLO  │
└─────────────────────────────┘   └─────────────────────────────┘   └──────────────────────────────┘
┌────────────────────────────────────── COMMON FOUNDATION ──────────────────────────────────────┐
│ real Controller + LocalClient + volumes  |  deterministic engine  |  one cost model and ledger│
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```
<!-- text-diagram:capabilities:end -->

`putget_sim` is the fixture: the real stack with no capability decision.
`dedup_sim` adds one source-routing decision and read-through replication.
`kvcache_sim` adds serving placement, prefix reuse, queue/decode sensors, SLO
admission, and a serving lifecycle. Their detailed scenarios and metrics remain in
their package READMEs.

## 5. Fidelity boundaries

- Controller mutation uses the real synchronous helpers; the small read endpoint
  bodies are mirrored because Monarch endpoint descriptors cannot be called
  off-actor.
- `create_transport_buffer` is a process-global TorchStore hook. One shared factory
  owns the substitution and resolves the calling client from task-local context.
- Costs are analytic properties of the target `MachineProfile`, not measurements of
  the machine running the simulation.
- The DES explores logical interleavings, not unsynchronized shared-memory races,
  kernel internals, or real RPC serialization jitter.

## 6. Run and verify

```bash
PYTHONPATH=. .venv/bin/python -m putget_sim
PYTHONPATH=. .venv/bin/python -m dedup_sim
PYTHONPATH=. .venv/bin/python -m kvcache_sim

PYTHONPATH=. .venv/bin/python -m pytest \
  realsim/tests sim_common/tests dedup_sim/tests kvcache_sim/tests -q
PYTHONPATH=. .venv/bin/python -m realsim.tools.check_contract
PYTHONPATH=. .venv/bin/python -m realsim.tools.check_structure
```

The guards assert byte-identical traces, invariants under seeded scheduling,
off-sim byte correctness on small real tensors, allocation-free large modeled
payloads, prediction/charge parity, and end-to-end demo execution.
