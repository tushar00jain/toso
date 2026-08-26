# Scaling TorchStore Weight Synchronization

At every RL iteration, trainer ranks publish a new model snapshot and generator ranks
read it. Both publication and lookup contribute to the cost of each weight update.

Two questions dominate:

1. How does controller work grow as model keys, tensor slices, publishers, and readers
   increase?
2. When several sources can serve the same bytes, how can readers avoid choosing the
   same source and overloading it?

There are two solution paths:

1. Reuse indexed tensor layouts across snapshot versions and select eligible sources
   by cost.
2. Precompute direct trainer-to-generator routes and redistribute weights within
   generator replica groups.

## 1. Workload: Qwen3.6-27B

The representative workload uses Qwen3.6-27B on two 8-GPU nodes:

| Item | Configuration |
| --- | ---: |
| Model | 64 layers, 1,199 checkpoint tensor keys, 55.6 GB |
| Trainer | 8 ranks: $\mathrm{FSDP}=4$, $\mathrm{TP}=2$ |
| Generator | 8 ranks: $\mathrm{DP}=2$, $\mathrm{TP}=4$; two model replicas |

Each iteration publishes one new snapshot from the trainer ranks and distributes it
to every generator rank. The calculations below use the checkpoint's 1,199 tensors as
a full-state upper bound; the runtime TorchStore state dict may omit runtime-specific
or unowned keys.

## 2. What scales poorly in TorchStore?

Let:

- $T$ be trainer ranks publishing the snapshot;
- $G$ be generator ranks reading the snapshot;
- $P_t$ be the `(key, volume, TensorSlice)` records published by trainer rank $t$;
- $Q_g$ be the keys requested by generator rank $g$;
- $V_k$ be the storage volumes recorded for key $k$;
- $S_k$ be the `TensorSlice` records stored for key $k$ across those volumes;
- $D_k$ be the number of dimensions in key $k$'s tensor ($D_k=2$ for a matrix).

### 2.1 Snapshot publication

The write path inserts or updates every published key and slice in the directory. Its
cost is $O(\sum_t \lvert P_t \rvert)$ across the trainer ranks. Replacing or retiring
those records has the same scale.

If every trainer rank publishes one slice for each checkpoint tensor, the example
workload creates $1{,}199 \times 8 = 9{,}592$ `(key, volume)` associations per
iteration. This is the worst case: eight volumes with up to 1,199 keys each.

Snapshot publication therefore scales linearly with the number of directory records
written. For centralized metadata that materializes each published record, this is the
necessary lower bound and is acceptable at the current example scale.

### 2.2 Lookup and reshard planning scan every source

A batched `get_state_dict` then uses one controller call, but that call still performs
work for every key:

```text
key trie -> every volume's StorageInfo -> compare every stored TensorSlice with the request
```

For each requested key $k$, the controller visits its $V_k$ volume records and
$S_k$ stored slices, while the client compares every slice across $D_k$ dimensions.
The exact burst cost is:

$$
O\left(\sum_g \sum_{k \in Q_g} \left(V_k + S_k D_k\right)\right)
$$

In the uniform case, every generator requests the same keys, every key is stored on a
number of volumes proportional to the trainer count, each volume contributes one
slice, and tensor dimensionality is bounded:

$$
Q_g = Q \quad \forall g,
\qquad
V_k = \Theta(T),\quad S_k = V_k,\quad D_k = O(1)
\quad \forall k \in Q.
$$

The burst cost becomes:

$$
O\left(G \lvert Q \rvert T\right)
$$

When trainer and generator counts grow together, $T=\Theta(G)$, this becomes
$O(\lvert Q \rvert G^2)$: quadratic in the number of participating ranks at fixed
model size.

Using the same full-key upper bound, one generator request can inspect up to 9,592
trainer-volume records, followed by the corresponding tensor-slice intersection
checks. Eight generator ranks can drive up to 76,736 such entry visits in one
synchronized refresh.

### 2.3 Deterministic source selection creates a hot source

Replicas do not spread traffic by themselves. In the worst case, deterministic
ordering sends one slice for every requested key to the same storage volume. That
volume serves the total number of key slices requested by all generators:

$$
\sum_{g=1}^{G}\lvert Q_g\rvert
$$

In the uniform case, every generator requests the same key set:

$$
Q_g = Q \quad \forall g.
$$

The hot-volume load therefore grows linearly with the generator count and the number
of keys in the shared request:

$$
O\left(G\lvert Q\rvert\right)
$$

Equivalent volumes remain idle, so source capacity does not scale with the number of
available replicas.

Equivalent volumes are storage volumes that can serve the same requested
`TensorSlice`. They exist when trainer DP ranks publish replicated slices. They could
also arise when a generator fetching a slice publishes its local copy so later
generators can read through it, but the current weight-sync path does not register
in-flight generator copies or route later reads through them.

## 3. Two solution paths

### Option A: reuse indexed layouts and select sources by cost

1. Index `TensorSlice` geometry independently from the volumes that currently hold
   it. The controller can then find overlapping regions and reuse reshard geometry
   across keys and snapshot versions with the same layout.
2. Choose between a trainer that already holds the slice and a generator that is
   fetching it and has promised to publish a copy. Rank them by assigned-request load
   and network-transfer cost, waiting if the selected generator is still pending.

For this comparison, use the uniform case from section 2: every generator requests
the same key set, every key is stored on a number of volumes proportional to the
trainer count, each volume contributes one slice, tensor dimensionality is bounded,
and trainer and generator counts grow together:

$$
Q_g = Q \quad \forall g,
\qquad
\lvert P_t\rvert = \lvert Q\rvert \quad \forall t,
\qquad
V_k = \Theta(T),\quad S_k = V_k,\quad D_k = O(1)
\quad \forall k \in Q,
\qquad
T = \Theta(G).
$$

For the table, $S = \max_{k \in Q} S_k$ is the largest number of `TensorSlice`
records stored for any requested key. The index contains at most $S$ distinct
`TensorSlice` geometries per key because replicas can store duplicate geometries.
Building the ordered index for a new layout costs $O(S\log S)$; later snapshots with
the same layout reuse it. Since $D_k=O(1)$, each candidate `TensorSlice` intersection
check costs $O(1)$. A source-cost index maintains the eligible volumes in cost order.
Adding a source or changing its readiness or assigned load costs $O(\log G)$, as does
selecting and updating a source. Across all $G$ generator batches, source selection
therefore costs $O(G\log G)$. The same index serves every key with the same placement,
so this cost is not multiplied by $\lvert Q\rvert$.

The table reports cluster-wide controller work: the snapshot must be published and
generator plans produced before the corresponding data transfers can proceed.

| Operation | Current path | Indexed path |
| --- | ---: | ---: |
| Snapshot publication | $O(G\lvert Q\rvert)$ | $O(G\lvert Q\rvert + S\log S + G\log G)$ |
| Lookup and reshard planning | $O(\lvert Q\rvert G^2)$ | $O(G\lvert Q\rvert\log S + G\log G)$ |
| Hot-volume serving load | $O(G\lvert Q\rvert)$ slice reads on one volume | Distribute those reads across eligible volumes |

#### Implementation

The trainer-publication and generator read-through flows are shown in the
[appendix](#appendix).

The geometry index maps each requested `TensorSlice` to a shared set of live trainer
and pending-generator volumes. Each source set maintains those volumes in cost order;
publication, routing, and completion events update it in $O(\log G)$.

The ordering uses a cost such as:

$$
\text{cost} = \text{readiness wait} + \text{transfer time}
              + \text{source load} + \text{fabric penalty}
$$

The selected source's load is recorded before the next decision. A live trainer can
serve immediately; selecting a pending generator installs a readiness gate that waits
for its local publication before returning the plan. The client then executes the plan
with ordinary `get` and `put` operations.

The same indexed, cost-based source selection can extend to KV-cache placement and
reuse.

### Option B: precompute direct transfer in the application

Exchange trainer and generator shard metadata once, compute direct sender-to-receiver
routes, and reuse them for every update. One generator per TP shard receives the
trainer data and redistributes it to the other DP replicas.

At each update, trainer ranks place the planned slices in their local transport
buffers. After a trainer-side barrier, trainer rank 0 sends the snapshot version to
the generator coordinator, which tells the elected generators to pull over the
precomputed routes. The generators redistribute the received slices within their DP
groups and install the new weights. TorchStore stores no per-update state and is not
called.

| Operation | Current path | Application-managed path |
| --- | ---: | ---: |
| Controller snapshot publication | $O(G\lvert Q\rvert)$ | $0$ per update |
| Controller lookup and reshard planning | $O(\lvert Q\rvert G^2)$ | $0$ per update |
| Trainer-to-generator slice deliveries | $G\lvert Q\rvert$ | $G\lvert Q\rvert/\mathrm{DP}$ |
| Generator-to-generator slice deliveries | $0$ | $G\lvert Q\rvert(\mathrm{DP}-1)/\mathrm{DP}$ |

For the Qwen3.6-27B configuration, four generators receive weights from trainer ranks,
and the other four receive them from those generators. Before updates begin, the
application creates an architecture-independent transfer plan from the trainer and
generator shard specifications. Each generator executes its assigned part of that
plan for every snapshot.

This path removes the TorchStore control plane from steady-state weight updates. The
application instead manages route setup and the transfer lifecycle. This remains a
weight-transfer subsystem rather than a general cache solution.

### Choosing between them

| Goal | Better fit |
| --- | --- |
| Improve general TorchStore reads and dynamic replicas | Indexed layouts + cost-based selection |
| Reuse TorchStore's normal publication and lookup path | Indexed layouts + cost-based selection |
| Remove per-update controller work for fixed weight routes | Application-managed direct transfer |
| Reduce trainer-origin traffic across DP replicas | Direct transfer + replica redistribution |

## Appendix

### Option A Read-Through Flow

Trainer publication:

```text
trainer put_state_dict
  -> local volume stores TensorSlices
  -> controller records the trainer volume
  -> geometry and source-cost indexes update
```

Generator read-through:

```text
generator get_state_dict
  -> register the generator's pending local copy
  -> find overlapping TensorSlices and select a source by cost
  -> wait if the selected generator source is still pending
  -> fetch and reshard into the generator
  -> store the local copy and mark its publication complete
```

The controller handles metadata only. Tensor bytes move directly between storage
volumes, and a completed generator read-through becomes a source for later reads.
