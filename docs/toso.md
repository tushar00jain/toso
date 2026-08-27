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
| Model | 64 layers, 1,199 checkpoint tensor keys, $M=55.6\ \mathrm{GB}$ |
| Trainer | 8 ranks: $\mathrm{FSDP}=4$, $\mathrm{TP}=2$ |
| Generator | 8 ranks: $\mathrm{DP}=2$, $\mathrm{TP}=4$; two model replicas |

Transfer-time estimates use:

- Trainer $\rightarrow$ generator: InfiniBand/RDMA at $25\ \mathrm{GB/s}$ per node.
- CPU-staged trainer $\rightarrow$ generator: $17.5\ \mathrm{GB/s}$ effective bandwidth.
- Generator $\rightarrow$ generator: NVLink/NVSwitch at $1.8\ \mathrm{TB/s}$ bidirectional, or $900\ \mathrm{GB/s}$ one way, per GPU.

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

**Uniform assumption.** Every generator requests the same key set, every trainer
publishes one slice for each key, every key is stored on a number of volumes
proportional to the trainer count, each volume contributes one slice, tensor
dimensionality is bounded, and trainer and generator counts grow together. The
generator ranks form DP replicas of TP-sharded models:

$$
Q_g = Q \quad \forall g,
\qquad
\lvert P_t\rvert = \lvert Q\rvert \quad \forall t,
\qquad
V_k = \Theta(T),\quad S_k = V_k,\quad D_k = O(1)
\quad \forall k \in Q,
\qquad
T = \Theta(G),
\qquad
G = \mathrm{TP}\,\mathrm{DP}.
$$

The data-plane load tables count one equal-cost logical slice per requested key.

### 2.1 Snapshot publication

The write path inserts or updates every published key and slice in the directory. Its
exact cost across the trainer ranks is:

$$
O\left(\sum_t \lvert P_t \rvert\right)
$$

Under the uniform assumptions, the publication cost becomes:

$$
O\left(G\lvert Q\rvert\right)
$$

Replacing or retiring those records has the same scale.

With 1,199 keys and eight trainer ranks, the example workload creates
$1{,}199 \times 8 = 9{,}592$ `(key, volume)` associations per iteration. This is the
worst case: eight volumes with up to 1,199 keys each.

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

Under the uniform assumptions, the burst cost becomes:

$$
O\left(\lvert Q \rvert G^2\right)
$$

With 1,199 keys, one generator request can inspect up to 9,592 trainer-volume records,
followed by the corresponding tensor-slice intersection checks. Eight generator ranks
can drive up to 76,736 such entry visits in one synchronized pull.

### 2.3 Deterministic source selection creates a hot source

In the current path, generators read their requested slices only from trainer ranks.

Under the uniform assumptions, the load is:

| Role | Count | Ingress per rank | Egress per rank |
| --- | ---: | ---: | ---: |
| Trainer rank | $\Theta(G)$ | $0$ | $\lvert Q\rvert$ on average |
| Fan-in/out generator | $G/\mathrm{DP}$ | $\lvert Q\rvert$ | $0$ |
| Other generator | $G(\mathrm{DP}-1)/\mathrm{DP}$ | $\lvert Q\rvert$ | $0$ |

The completion time is:

$$
\frac{\mathrm{DP}\,M}{17.5\ \mathrm{GB/s}}
= \frac{2(55.6\ \mathrm{GB})}{17.5\ \mathrm{GB/s}}
= 6.354\ \mathrm{s}.
$$

The trainer ranks serve all $G\lvert Q\rvert$ slices, while generators that could
relay fetched slices remain idle.

## 3. Two solution paths

### Option A: change the per-key index and select sources by cost

Change the directory layout from:

```text
key -> volume -> TensorSlice
```

to:

```text
key -> indexed TensorSlice geometries -> source volumes ordered by cost
```

1. Indexed `TensorSlice` geometries return only slices that overlap the request.
2. A completed generator becomes a read-through cache later reads.
3. Each matched geometry has a list of live trainer and pending-generator sources
   ordered by the cost function defined below.

The ordering uses a cost such as:

$$
\text{cost} = \text{readiness wait} + \text{transfer time}
              + \text{source load} + \text{fabric penalty}
$$

The selected source's load is recorded before the next decision. A live trainer can
serve immediately. If a pending generator is selected, the requester waits until that
generator has the data.

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

Let $S = \max_{k \in Q} S_k$. Building the geometry index for a new layout costs
$O(S\log S)$; later snapshots reuse it. The interval index takes $O(\log S)$ to
locate candidates, so each of the $G\lvert Q\rvert$ key lookups costs $O(\log S)$.
Selecting or updating one source in the ordered list costs $O(\log G)$.
With a bounded number of selected sources for each of the $G$ requests, this adds
$O(G\log G)$.

The snapshot must be published and generator plans produced before the corresponding
data transfers can proceed:

| Operation | Current path | Indexed path |
| --- | ---: | ---: |
| Snapshot publication | $O(G\lvert Q\rvert)$ | $O(G\lvert Q\rvert + G\log G)$ |
| Lookup and reshard planning | $O(\lvert Q\rvert G^2)$ | $O(G\lvert Q\rvert\log S + G\log G)$ |

One generator per TP shard fetches from the trainers, and the other DP replicas read
through it. The resulting data-plane load is:

| Role | Count | Ingress per rank | Egress per rank |
| --- | ---: | ---: | ---: |
| Trainer rank | $\Theta(G)$ | $0$ | $\lvert Q\rvert/\mathrm{DP}$ on average |
| Fan-in/out generator | $G/\mathrm{DP}$ | $\lvert Q\rvert$ | $\lvert Q\rvert(\mathrm{DP}-1)$ |
| Other generator | $G(\mathrm{DP}-1)/\mathrm{DP}$ | $\lvert Q\rvert$ | $0$ |

The completion time is:

$$
\frac{M}{17.5\ \mathrm{GB/s}}
  + \frac{M/\mathrm{TP}}{900\ \mathrm{GB/s}}
= 3.177\ \mathrm{s} + 0.015\ \mathrm{s}
\approx 3.193\ \mathrm{s}.
$$

Callers only invoke `get`. The same mechanism can be reused as part of a KV-cache implementation.

### Option B: precompute direct transfer in the application

Build a route table once from the trainer and generator `TensorSlice` layouts:

```text
generator TensorSlice -> overlapping trainer TensorSlices -> fixed transfer routes
```

1. The shard geometry determines the source and destination byte ranges.
2. When several trainer volumes hold the same slice, assign one during setup to
   balance trainer egress.
3. For each replicated generator slice, route one copy from the trainers and the
   remaining copies from that generator to its DP peers.

No dynamic cost function is needed: sharding determines the routes, and trainer load
is balanced once during setup. The route table is distributed; each trainer and
generator stores only its local send and receive entries. There is no per-update
controller lookup, route computation, or TorchStore call; each rank traverses its
saved local routes. Each generator executes its assigned part of the saved plan for every
snapshot:

| Operation | Current path | Application-managed path |
| --- | ---: | ---: |
| Controller snapshot publication | $O(G\lvert Q\rvert)$ | $0$ per update |
| Route selection and reshard planning | $O(\lvert Q\rvert G^2)$ controller work | $O(\lvert Q\rvert)$ local work per rank |

The resulting data-plane load is:

| Role | Count | Ingress per rank | Egress per rank |
| --- | ---: | ---: | ---: |
| Trainer rank | $\Theta(G)$ | $0$ | $\lvert Q\rvert/\mathrm{DP}$ on average |
| Fan-in/out generator | $G/\mathrm{DP}$ | $\lvert Q\rvert$ | $\lvert Q\rvert(\mathrm{DP}-1)$ |
| Other generator | $G(\mathrm{DP}-1)/\mathrm{DP}$ | $\lvert Q\rvert$ | $0$ |

The transfer time is the same as Option A: approximately $3.193\ \mathrm{s}$.

This path removes the TorchStore control plane. This remains a weight-transfer subsystem
rather than a general cache solution.
