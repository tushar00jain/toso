# `realsim` — a real-code cooperative DES for TorchStore

`realsim` is a single-threaded, deterministic discrete-event simulation (DES)
that drives the **real** TorchStore client planning core, the **real** controller
directory, and the **real** in-memory transport/store off-actor, under a virtual
clock. It models only the *new* read coordinator — the one component that does not
exist in production yet — behind a pluggable policy seam.

`realsim` is the real-code foundation the two capability simulations build on:
both [`dedup_sim/`](../dedup_sim/) and [`kvcache_sim/`](../kvcache_sim/) `import
realsim` and run their algorithms against the real directory and the real
TorchStore types. It deliberately depends on the **real `torchstore` / `torch` /
`monarch` install** (the from-source build in the repo `.venv`); it is *not*
stdlib-only. That dependency is the point: the client, controller directory,
transport, and store types that execute are the production ones. Only the
coordinator (the component being designed) and the actor/RPC boundary are
substituted with in-process seams.

This document is the single design reference for `realsim`: the concurrency model
that makes it sound, the architecture, exactly how each real object is driven
off-actor, the cost model and allocation-free data plane, the coordinator and its
policy seam, the CI-enforced concurrency contract, and how the capability sims
consume it. For a narrative walk-through of the whole DES foundation and how the
two capability sims differ, see [`des_explained.md`](des_explained.md).

---

## 1. Goal and non-negotiables

Run the **real, already-built** TorchStore components — the client (`ts.get` /
`ts.put`) and the storage-volume controller/directory — under a deterministic DES,
and keep only the *new* coordinator as a model. This tests real code paths for the
mature pieces while still iterating fast on the unbuilt algorithm.

The whole point is to test what production does, so the design holds a hard line:

- **Drive real code; never reimplement TorchStore logic.** Subclass/reuse the real
  `TransportBuffer` ABC, and reuse the real `InMemoryStore`, `Controller`, and
  `LocalClient` planning methods. What executes must be the real code.
- **No torchstore edits.** Drive it as-is. Where a small upstream change would
  close a fidelity gap, it is recorded as a recommendation (§10), not made.
- **Real types only.** The directory is the real `Controller`; lookups return real
  `StorageInfo` / `TensorSlice`; puts use real `Request`s. There are no invented
  stand-in types and no encode/decode translation layer between the sim and
  TorchStore.
- **One clock, via the loop.** Read time with `asyncio.get_running_loop().time()`
  (virtual under the engine, real under a plain asyncio loop). No separate clock
  module.
- **Reuse `sim_common`.** `sim_common/trace.py` for all tracing;
  `sim_common/cost_model.py` for every resource cost;
  `sim_common/topology` for locality identity; `sim_common/report.render_tree`
  for the source→dest picture. No parallel utilities.
- **Determinism.** Same inputs ⇒ byte-identical trace. No wall-clock reads in
  control flow, no threads, no `os.fork`/`multiprocessing`, no blocking
  `time.sleep`, no unseeded randomness on the sim path. `asyncio.sleep` on the
  loop's virtual clock is the sanctioned way to advance time. Enforced by the
  concurrency-contract lint (§9).

---

## 2. Why a single-threaded cooperative DES is a *sound* model

Whether real code can run under a simulator is not about how clever the engine is
— it is about **where the abstraction seam sits**. If you reimplement a
component's logic against the simulator (seam at the top), the code you ship is not
the code you tested. If instead the component runs unchanged and the simulator
virtualizes the two things application code unavoidably touches — **time** and
**I/O / transport** — *beneath* it (seam at the bottom), the code is oblivious to
being simulated. `realsim` uses the bottom seam via a **functional-core / sans-IO**
split: TorchStore's pure planning and directory logic is driven directly, and its
actor/endpoint/transport wrapper is replaced with in-process seams.

### 2.1 Two classes of concurrency bug

"Race condition" hides two different bugs that need different tools:

- **Class A — logical / interleaving races.** The order in which messages, RPC
  completions, timers, and resumptions interleave (a crash between "decided" and
  "persisted"; a stale read across an `await`). These are the bugs a coordinator
  lives or dies by.
- **Class B — true shared-memory data races.** Two OS threads touch the same
  memory unordered (a missing lock, a torn read) — undefined behavior.

A single-threaded DES finds class A and **cannot represent class B at all** (on one
thread every access is totally ordered; switches happen only at yield points, never
mid-instruction).

### 2.2 Data-race-freedom makes the single-thread DES sound

Rather than *detect* class B, `realsim` makes it **unrepresentable**. A
single-threaded cooperative DES is a **sound and complete** model of a program's
concurrency **iff the program is data-race-free (DRF)** — under DRF, observable
behavior equals interleavings of synchronization operations, which is exactly what
the DES enumerates. DRF holds by construction under a **concurrency contract**:

1. **Share-nothing components.** A component's mutable state is touched only by its
   own task, serially; cross-component interaction is only messages over a
   transport the simulator mediates.
2. **Cooperative synchronization only.** Coordinate with primitives that suspend
   the *task* through the runtime (`asyncio` awaitables / async locks / queues),
   never primitives that block the OS thread or spawn threads. A cooperative
   primitive's contention is a scheduling point the DES sees and can randomize; a
   thread-blocking one is invisible to a single-thread sim.
3. **Share-nothing parallelism is fine.** True parallelism with no shared mutable
   state is observationally equivalent to a cooperative interleaving, so the
   single-thread DES stays a sound model. Delete *shared mutable state*, not
   parallelism.

TorchStore already has this shape (actors own their state; interaction is via RPC;
concurrency is `asyncio`, not threads), which is what makes the whole approach
work. The class-A/class-B problem then collapses into one: class B is engineered
out, and class A is fully covered by one deterministic, real-code, reproducible
simulator.

### 2.3 Enforce the seam — don't trust it

The recurring failure mode of "run real code" simulators is a seam that is
**opt-in rather than enforced**: a wall-clock call or a thread-blocking lock runs
fine, silently escapes the model, and determinism rots. `realsim` therefore
**guards the time and concurrency seams in CI** (§9), rather than merely
documenting them.

---

## 3. Architecture and package layout

```
realsim/
  __init__.py
  README.md                   # short how-to-run; points here
  seams/                      # in-process adapters onto real torchstore surfaces
    transport.py              # InMemoryTransport (subclasses real MonarchRPCTransportBuffer);
                              #   charges the cost model per put/get; re-exports Endpoint;
                              #   TensorDescriptor (metadata-only carrier)
    volume_handle.py          # FakeVolumeHandle (mirrors StorageVolume endpoints; real InMemoryStore)
    controller_handle.py      # FakeControllerHandle (dispatches to a real Controller)
    factory.py                # THE create_transport_buffer substitution point + source contextvar
  adapters/                   # thin wiring that constructs the real objects off-actor
    real_controller.py        # RealControllerAdapter (constructs a real Controller off-actor)
    real_client.py            # RealClientAdapter + FakeStrategy (single-client transport install)
  mesh.py                     # Mesh — multi-client wiring: per-node volumes + real clients,
                              #   one directory + registry, one shared transport factory
  coordinator/                # the NEW component under design — a model
    model.py                  # Reader, ReadPolicy/NaivePolicy, ReadCoordinator, BurstMetrics
  scenarios/                  # runnable scenarios
    burst_get.py              # synchronized read burst; meta/metadata data plane +
                              #   compute/network/storage/RAM cost exercise
  run_realsim.py              # `python -m realsim.run_realsim` demo entrypoint (+ --mode)
  tools/
    check_contract.py         # concurrency-contract lint (AST checker + CLI)
  tests/
    test_seams.py             # smoke: put + full get + sliced get round-trip
    test_determinism.py       # byte-identical traces; shape/dtype/nbytes invariants
    test_contract.py          # runs the lint + proves it detects each banned pattern
    test_correctness.py       # off-sim byte-equality on tiny REAL CPU tensors
    test_perf.py              # no-real-allocation-at-scale + parity vs. a capability sim
    test_composability.py     # import the real-directory backend + swap proof
    test_mesh.py              # Mesh wiring, per-operation source locality, one-owner install

sim_common/                   # shared DES library (repo root)
  async_engine.py             # deterministic asyncio loop + virtual clock
  cost_model.py               # MachineProfile + analytic network/RAM/storage/CPU/GPU costs
  engine.py, trace.py, topology.py, report.py   # reused as-is
  tests/test_async_engine.py  # determinism / virtual-time / gather-ordering
  tests/test_cost_model.py    # cost-model arithmetic + determinism
```

**What actually executes (real code):** the entire `LocalClient` planning core
(`_build_volume_requests`, `_expand_tensor_slices`, the `asyncio.gather` fan-out,
`_assemble_results`, `_apply_inplace`, `put_batch`); the `Controller` directory
state and its `_notify_put` / `_notify_delete` / `_is_dtensor_fully_committed`
logic; the entire `MonarchRPCTransportBuffer` + base `TransportBuffer` lifecycle;
the entire `InMemoryStore`; and the real `Request` / `TensorSlice` / `StorageInfo`
/ `Trie` types.

**What is modeled or glue:** the new `ReadCoordinator` and its `ReadPolicy` (the
component being designed); the `.call` / `.call_one` awaitable wrappers standing in
for Monarch RPC; the ~5-line verbatim mirrors of the `locate_volumes` / `keys`
read endpoints (see §4); the transport's resource-cost charges; the minimal
`FakeStrategy`; the `Mesh`/`Cluster` wiring that constructs the real objects; and
the `create_transport_buffer` substitution.

---

## 4. How each real object is driven off-actor

The coupling between TorchStore and Monarch sits at the `@endpoint` +
`.call()/.call_one()` boundary, **not** in the algorithms. Everything below was
verified against the sibling `../torchstore` checkout and drives it **without
modifying any torchstore source**.

### `Controller` (`torchstore/controller.py`)

- **Construction.** `Controller()` constructs fine off-actor: it subclasses
  `monarch.actor.Actor`, but `__init__` only sets plain attributes (a `Trie`
  directory + flags) and never touches Monarch.
- **Initialization.** The real `init` `@endpoint` needs a live Monarch
  storage-volume mesh, but the directory logic only needs `is_initialized` (which
  `init` sets last), so `RealControllerAdapter` sets `controller.is_initialized =
  True` directly, mirroring the tail of `init`.
- **Endpoints are not directly callable off-actor.** `controller.locate_volumes`,
  `.notify_put_batch`, `.keys` are `EndpointProperty` descriptors that expose no
  call surface off-actor, so the handle cannot `await` them directly.
- **How `FakeControllerHandle` drives them.** `notify_put_batch` /
  `notify_delete*` are already plain sync helpers upstream
  (`Controller._notify_put` / `_notify_delete`), so the handle calls the **real**
  helper (after `assert_initialized`). The `locate_volumes` / `keys` read bodies
  are *not* extracted into sync helpers upstream, so the handle **mirrors the ~5
  line endpoint bodies verbatim** (quoted in comments). Those mirrors still touch
  only the real object's state (`controller.keys_to_storage_volumes` over the real
  `Trie`, `controller._is_dtensor_fully_committed`), so the directory *state and
  semantics* are 100% real; only the trie-read glue is restated. (Extracting those
  two bodies upstream would close this last gap — §10.)

### `LocalClient` (`torchstore/client.py`)

- **Construction.** `LocalClient(controller_handle, strategy)` — no Monarch. The
  real planning core runs unmodified.
- **Controller RPC seam.** The client calls
  `self._controller.locate_volumes.call_one(...)`, `.notify_put_batch.call(...)`,
  `.keys.call_one(...)`. `FakeControllerHandle` presents `.call` / `.call_one`
  awaitables on each name; in-process both resolve to the same coroutine, and
  callers that ignore a real `.call`'s `ValueMesh` return are unaffected.
- **Transport seam (the one substitution).** The client resolves the transport via
  the module global `create_transport_buffer`, imported at module load. The only
  substitution point is that bound name on the client module object, and
  `seams/factory.py` is the only place in the repo that touches it: it patches
  `sys.modules["torchstore.client"].create_transport_buffer` for the scope of a
  block and restores it in `finally`. (The submodule must be fetched via
  `sys.modules["torchstore.client"]` because the package namespace shadows the
  `client` submodule with a `client` function — §10.) Because the binding is
  process-wide, `factory.installed()` permits **one owner at a time** and raises on
  an overlapping install; a second install would otherwise silently win while the
  first owner kept recording metrics, charging transfers the wrong source
  locality. `RealClientAdapter.installed()` is the single-client form (source
  pinned to that adapter's node); `Mesh.installed()` is the multi-client form
  (source resolved per operation from the factory's contextvar).
- **Strategy seam.** `FakeStrategy` supplies only what the client uses —
  `select_storage_volume()`, `get_storage_volume(volume_id)` (both returning real
  `StorageVolumeRef` objects), and `transport_context`. A real
  `TorchStoreStrategy` is avoided because its `set_storage_volumes` needs a live
  Monarch mesh.

### Transport (`torchstore/transport/*`)

- `InMemoryTransport` **subclasses the real `MonarchRPCTransportBuffer`**, so the
  whole real transport lifecycle executes: base
  `TransportBuffer.put_to_storage_volume` / `get_from_storage_volume`, and
  MonarchRPC's `_pre_put_hook`, `_pre_get_hook`, `handle_put_request`,
  `handle_get_request`, `_handle_storage_volume_response`, `drop`. MonarchRPC
  (not RDMA/gloo/shm) is used because it already moves data as plain in-process
  Python references — ideal for an in-memory sim: the buffer object handed to the
  volume *is* the object the client reads back, so nothing is serialized.
- The subclass overrides only the two public entry points to add virtual-clock
  resource costs (§7) via `asyncio.sleep` against `loop.time()`, recording each
  charge into the shared trace.

### Volume (`torchstore/storage_volume.py`)

- `FakeVolumeHandle` presents `.put.call` / `.get.call_one` / `.handshake.call_one`
  (plus delete/reset), each body **mirroring the real `StorageVolume` endpoint
  verbatim** — every real endpoint just delegates to `self.store`. The backing
  store is the **real `InMemoryStore`**, so `InMemoryStore.put` / `.get` /
  `_get_data` / `_extract_slice_from_tensor` / `_get_sharded_tensor` are what
  execute (full-tensor and sliced gets both exercised).

---

## 5. The deterministic engine (`sim_common/async_engine.py`)

`AsyncEngine` is a deterministic cooperative `asyncio` event loop, exposed via
`run_sim(coro, *, random_seed=None, trace=None)`:

- **Virtual clock.** `loop.time()` returns simulated time; `call_at` / `call_later`
  and `asyncio.sleep` schedule against it. Time advances only when the ready queue
  drains, jumping to the earliest pending timer (classic DES advance) — so `await
  asyncio.sleep(10)` costs ~0 wall-clock. Real `async` torchstore client and
  coordinator code runs under this loop unmodified.
- **Deterministic ready queue.** FIFO by insertion, with a `(time, seq)` heap of
  timers whose monotonic `seq` breaks ties, so the run is totally ordered and
  reproducible. An optional `random_seed` switches to a seeded per-tick shuffle for
  class-A interleaving sweeps (same seed ⇒ same order).
- **Trace.** Emits `sim_common/trace.py` rows for scheduling events so a run yields
  a stable, inspectable trace.

`sim_common/engine.py` (`Sim` + `Promise`) is the ancestor callback DES that
`AsyncEngine` grew from; they share the `(time, seq)` tie-break convention. It is
not on the sim path (the sims are all async) but is kept as the reference callback
DES.

---

## 6. Cost model (`sim_common/cost_model.py`)

A DES must **model every cost from a target-machine profile, never measure it on
the box running the sim** — measuring couples the sim to the test host and
misrepresents the production hardware. `cost_model.py` is a sibling to
`topology.py`: every cost is a deterministic function of a **modeled quantity**
(`nbytes` / `flops`) and a caller-supplied **`MachineProfile`** of target-hardware
constants.

- **`MachineProfile`** fields: per-`Tier` network `(latency, bandwidth)` (reusing
  `topology.Tier`); RAM (`ram_bandwidth`, `ram_latency`); storage
  (`storage_read_bw`, `storage_write_bw`, `storage_latency`); compute (`gpu_flops`
  per-dtype + `gpu_flops_default`, `gpu_mem_bandwidth`, `cpu_flops`); optional
  host↔device (`h2d_bandwidth`, `d2h_bandwidth`).
- **Cost functions**, each `quantity × profile → time` (returning `0.0` for a zero
  quantity or a same-endpoint transfer):
  - `network_time(src, dst, nbytes, profile)` — wraps, not forks,
    `topology.transfer_time`; the per-tier constants come from the profile.
  - `mem_copy_time(nbytes, profile)`.
  - `storage_time(nbytes, kind, profile)` for `kind ∈ {read, write}`.
  - `compute_time(flops, dtype, device, profile)` — a **roofline**:
    `max(flops / effective_flops, nbytes / mem_bw)`, the device selecting GPU vs.
    host rates.
- Pure arithmetic — no clocks, threads, RNG, or measurement — so it passes the
  lint. `DEFAULT_PROFILE` is an **illustrative** demo profile (plausible relative
  magnitudes, *not measured*); real callers supply their own from scenario config.
  `topology.transfer_time` stays as the network special case; `cost_model`
  generalizes it. All sims charge every duration through this one model, so there
  is a single hardware story across the repo.

---

## 7. Allocation-free data plane and the full resource exercise

The sim carries **zero real tensor bytes** regardless of modeled payload size, and
charges **every** resource analytically.

### Data plane — two allocation-free carriers (`--mode`)

- **`meta` (default).** The payload `W` is `torch.empty(n, dtype, device="meta")`
  — a **real** `torch.Tensor` with zero storage but exact `shape` / `dtype` /
  `nbytes`. It passes every `isinstance` / `shape` / `dtype` / `is_contiguous`
  check in the real torchstore path and is passed by reference through the fake
  volume handle, so a 256 MiB *modeled* payload allocates nothing.
- **`metadata`.** No tensor at all: a `TensorDescriptor(shape, dtype)` stands in
  for the payload (`tensor_val is None`). It is handed to `client.put` as an
  arbitrary object so it round-trips the real *object* put/get path (sidestepping
  the `put_batch` value-typing behavior in `client.py`); `_nbytes` reads the
  modeled size off the descriptor.

`FakeTensorMode` and Monarch's fake tensors are deliberately avoided (ambient-mode
fragility and background threads would trip the contract lint).

### Full resource exercise

One burst charges **all** resources through the cost model off one
`MachineProfile`, threaded via `profile=` uniformly through the producer, readers,
and coordinator:

- **compute / GPU** — the producer's generate step (`compute_time` on
  `compute_device`, default `"cuda"`), charged by the scenario before the put;
- **network** — the client↔volume fabric, charged in the transport seam;
- **storage** — a write on put, a read on serve, charged in the transport seam;
- **RAM** — host-memory staging on serve (`mem_copy`), charged in the transport
  seam.

So a **put** charges `network` (client→volume) + `storage write`; a **get** charges
`storage read` + `mem_copy` (host staging) + `network` (volume→client).

---

## 8. Mesh, coordinator and scenario

### `mesh.py` — the shared multi-client wiring

Before a scenario can express any capability it needs the same set of real
objects: a controller adapter, a `FakeVolumeHandle` per node, a
`RealClientAdapter` (hence a real `LocalClient`) per node, one shared
`ResourceRegistry`, and — because `create_transport_buffer` is a process-wide
global — *one* transport factory shared across clients that resolves the caller's
source endpoint dynamically. `Mesh` is that assembly:

```python
mesh = Mesh(topology, profile=profile, trace=trace)
with mesh.installed():
    mesh.bind_source("s0")
    await mesh.client("s0").put_batch(...)
```

`Mesh.on_transfer` is an optional `(kind, src_id, dst_id, nbytes, cost)` hook read
at call time, so a consumer built *after* the mesh can claim it for accounting.

This wiring is independent of the capability under test, so it does not belong to
any one of them. It originally lived inside `ReadCoordinator`, whose shape is a
single synchronized burst (`run_burst(readers, key)`); a capability that is not a
burst — `kvcache_sim`'s continuous arrival stream — could not reuse the
coordinator and re-derived the wiring underneath it, duplicating the factory, the
contextvar, and the per-node adapter construction. With `Mesh` extracted, the
coordinator is a burst-shaped consumer of a mesh and `kvcache_sim`'s `Cluster` is a
KV-shaped one (four directory verbs), so each capability package holds only
capability code.

### `coordinator/model.py` — the new component

- `ReadCoordinator.run_burst(readers, key)` consults the **real** controller
  directory (`locate_volumes`) then fans each reader's **real** `client.get` out on
  the engine via `asyncio.gather`. The real objects it drives come from a
  `Mesh` (§8a); the coordinator adds only the read path, the policy seam, and the
  fabric accounting. It installs the mesh's shared, contextvar-aware
  `create_transport_buffer` for the burst so concurrent readers each charge the
  right locality: the process-wide transport global is replaced by one factory that
  reads the calling reader's source endpoint from a `ContextVar` set per reader
  task (`asyncio` copies the context into each `gather`-created task, so the lookup
  is task-local and deterministic — §10 recommendation 2 worked around).
- `ReadPolicy` is the pluggable seam. `NaivePolicy` (shipped) fetches
  independently: in a synchronized burst every reader locates the origin before
  anyone finishes, so each pulls from the origin volume — `m×` fabric, the
  baseline. A dedup/cache-aware policy overrides `run_burst` (to stage the burst
  into a read-through chain/tree) and/or `after_fetch` (to register a finished
  reader back into the **real** directory via `notify_put_batch`, the real
  read-through path, so later `locate_volumes` calls route to a closer peer). The
  routing is expressed purely by mutating real directory state; `dedup_sim`'s
  `DedupPolicy` is exactly such an override.
- `BurstMetrics` accounts `fabric_bytes` (origin-served) vs `total_get_bytes` and
  records `(src, dst, key)` edges for `render_tree`. For the naive policy the two
  are equal (`m×`); a dedup policy drives `fabric_bytes` toward the 1× union while
  `total_get_bytes` stays `m×`.

### `scenarios/burst_get.py` — a synchronized read burst

One origin volume on node `P` holds `W`; `m` reader volumes on distinct hosts of
node `R` each want overlapping data (default: the whole tensor). `build_burst(...)`
/ `run_burst(...)` seed `W`, run the burst on a fresh engine, and return a
`BurstResult` (trace, metrics, results, expected carrier). They take `mode=`
(`"meta"` default / `"metadata"`, §7), `profile=` (the target `MachineProfile`,
§6), and `compute_device=` (the producer's roofline device, default `"cuda"`).
`render_burst_summary` prints the fabric/wallclock digest + the ASCII source→dest
tree.

---

## 9. Concurrency-contract lint (`realsim/tools/check_contract.py`)

A dependency-free AST checker that fails the build if any **simulated code path**
under `realsim/` or `sim_common/` reaches for a determinism-breaking primitive. It
is scoped to those two directories only — the sibling `../torchstore` is not
scanned.

**Banned on the sim path:**

- `threading` / `_thread` / `multiprocessing` imports or use;
- `os.fork` / `os.forkpty`;
- `time.sleep` (blocking wall-clock sleep);
- wall-clock **reads** in library code — `time.time` / `perf_counter` / `monotonic`
  (and `*_ns` forms);
- unseeded randomness — module-global `random.<fn>()`, unseeded `random.Random()`,
  or `random.SystemRandom`.

**Explicitly allowed:**

- `asyncio.sleep` — the sanctioned way to advance the loop's virtual clock;
- `random.Random(seed)` with an explicit seed;
- wall-clock **reads inside test modules** (`tests/` or `test_*.py`) — those
  measure elapsed wall time only to *assert* the virtual clock is free (e.g.
  `test_async_engine.py` proves `asyncio.sleep(10)` costs ~0s); never control flow.

**Out of scope, benign:** `torchstore/logging.py::LatencyTracker` uses
`perf_counter()` for DEBUG-only elapsed display — it never affects control flow or
the trace, and it lives in `../torchstore`, which the lint does not scan.

The lint runs standalone and as a test (`tests/test_contract.py`), which also
proves the checker detects each banned pattern and passes the sanctioned ones — a
green run means the contract holds, not that the lint is asleep.

---

## 10. Recommendations to torchstore

None of these are made (the sim drives torchstore as-is); they are the small
upstream changes that would each remove a piece of glue:

1. **Extract the `locate_volumes` and `keys` `@endpoint` bodies into plain sync
   helpers** (e.g. `_locate_volumes` / `_keys`, mirroring the existing
   `_notify_put`). Then the handle would call **real** code instead of mirroring
   ~5 lines, closing the last fidelity gap in the directory read path.
2. **Make the transport factory injectable per-client** instead of the module
   global `create_transport_buffer` — e.g. accept an optional factory on
   `LocalClient`, or hang it off the strategy. This removes the process-wide
   substitution and is the clean form of driving many clients with different
   transports in one process. It would also let `Mesh` drop its contextvar
   (each client would simply hold its own factory) and remove the one-owner
   restriction in `seams/factory.py` entirely.
3. **Add a non-actor `Controller` construction/init path** — a plain constructor
   plus an init that does not require a Monarch storage-volume mesh (or split
   directory init from mesh setup). Today the adapter sets `is_initialized`
   directly, which reaches past the intended API.
4. **Minor:** `torchstore.client` (the submodule) is shadowed by a `client`
   function on the package namespace, so patchers must use
   `sys.modules["torchstore.client"]`. Renaming the factory function or the
   submodule would remove this footgun.

---

## 11. Composability — the capability sims are real consumers

The **real controller directory** is a cleanly separable, importable unit: `from
realsim.adapters.real_controller import RealControllerAdapter` (and
`FakeControllerHandle`) is a plain top-level import with **no import-time side
effects** (no background threads, no running event loop, no Monarch mesh), and its
directory operations (`notify_put_batch` → `locate_volumes` / `keys`) run
standalone. `tests/test_composability.py` proves the import and the
empty-on-absent lookup contract.

Both capability sims consume `realsim` directly, speaking the real torchstore
`Request` / `TensorSlice` / `StorageInfo` / `Trie` types natively — so there is no
region↔`TensorSlice` translation layer anywhere:

- [`dedup_sim/`](../dedup_sim/) implements dedup routing as a real
  `ReadPolicy` (`DedupPolicy`) plugged into the `ReadCoordinator`, driving the real
  `Controller` directory + `LocalClient` to a 1× peer read-through.
- [`kvcache_sim/`](../kvcache_sim/) consults the real `Controller` directory for
  KV-block presence (`Cluster.prefix_lengths` over `locate_volumes`) and drives
  real per-instance clients for prefix publish / remote pull / eviction.

`test_composability.py` also builds a tiny 1-D region↔`TensorSlice` round-trip, but
only to *document* how the real directory speaks slice types — it is the
translation the capability sims do not need, since they speak those types directly.

---

## 12. Correctness and performance guards

- **Correctness (`tests/test_correctness.py`, off-sim).** The DES path is
  allocation-free, so it asserts only `shape` / `dtype` / `nbytes` + trace
  determinism. Byte-level reassembly correctness is checked off the engine: tiny
  **real** CPU tensors put/got through the *same* real client / controller /
  `InMemoryStore` code (a plain asyncio loop), asserting exact bytes for a full and
  a sliced get.
- **No real allocation at scale (`tests/test_perf.py`).** A 256 MiB *modeled*
  payload must not move peak RSS, and every carrier has a null data pointer or is a
  descriptor.
- **Not materially more expensive than a capability sim (`tests/test_perf.py`).** A
  `realsim` run and a `dedup_sim` run are each measured in a fresh subprocess
  (import-dominated end-to-end wall + RSS), and `realsim` must stay within a
  tolerant multiple of the capability sim (both share the torch/monarch import
  baseline). Wall-clock reads here are assertion measurement (allowed in test
  modules), never sim-path control flow.

---

## 13. Running

Use the venv interpreter that has torchstore/torch/monarch (see the repo-root
[`README.md`](../README.md) "Building the live example from source"). Run from the
repo root with `PYTHONPATH` set to it; pass the test dirs explicitly (there is no
`pytest.ini`).

**The whole cross-package suite** (realsim + shared engine + both capability sims):

```
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m pytest \
  realsim/tests sim_common/tests dedup_sim/tests kvcache_sim/tests -q
```

**The concurrency-contract lint** (also covered by `test_contract.py`):

```
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.tools.check_contract
```

**The demo:**

```
PYTHONPATH=<repo-root> <repo-root>/.venv/bin/python -m realsim.run_realsim \
  [-m READERS] [-n N] [--mode meta|metadata] [--seed S] [-v]
```

- `-m/--readers N` — readers in the burst (default 3).
- `-n/--elements N` — elements in `W` (float32; payload = `4*N` bytes).
- `--mode meta|metadata` — the allocation-free data-plane carrier (§7).
- `--seed S` — switch the engine to seeded-random ready-queue mode (default: FIFO,
  reproducible).
- `-v` — also print the full per-event virtual-time trace (DEBUG).

It prints the fabric/wallclock summary + the ASCII source→dest tree at INFO. Under
the naive policy every reader pulls the origin (`m×` fabric) — the baseline a
read-through policy would cut toward the 1× union. Costs come from the target
`MachineProfile` (the illustrative `DEFAULT_PROFILE` in the demo), never measured
on the box running the sim.

---

## 14. Non-goals

- No changes to `../torchstore` (recommend, don't edit — §10).
- No real network/RDMA/disk; no multi-process; no Monarch actor spawning.
- No real GPU/collective execution; the control path stays metadata-only and every
  cost is analytic.
- The new coordinator stays a model until its interface stabilizes; only then would
  extracting it as real code be worthwhile.
