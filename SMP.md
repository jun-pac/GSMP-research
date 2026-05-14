
## Prompt for Claude

You are an expert PyTorch Geometric / DGL / graph ML engineer.

I want you to write clean, modular code that applies **Symmetrized Message Passing (SMP)** to baseline GNN models.

### Background

I am working with temporal node-classification datasets where each node has a timestamp. The goal is to reduce time-dependent bias in message passing by modifying the graph aggregation weights.

The method is called **Symmetrized Message Passing (SMP)**.

Conventional message passing averages neighbor features:

[
M_v^{(k+1)}
===========

\frac{
\sum_{\tilde y}
\sum_{\tilde t}
\sum_{w\in \mathcal{N}_v(\tilde y,\tilde t)}
X_w^{(k)}
}{
|\mathcal{N}_v|
}.
]

However, because the temporal neighborhood around a target node can be asymmetric near the temporal boundary, the first moment of the aggregated message can depend on the target timestamp (t).

SMP fixes this by doubling the contribution of “single” temporal neighbors.

For a target node (v) with timestamp (t=time(v)), define:

[
\Delta = |time(u)-time(v)|.
]

Let

[
r_v = \min(t_{\max} - time(v),; time(v)-t_{\min}).
]

For an edge (u \to v), node (u) is treated as a **single node** relative to target (v) when

[
|time(u)-time(v)| > r_v.
]

In that case, its message weight should be doubled.

Equivalently, SMP reconstructs the graph by modifying the adjacency:

[
A'_{uv}
=======

\begin{cases}
2A_{uv}, & |time(u)-time(v)| > \min(t_{\max}-time(v), time(v)-t_{\min}),\
A_{uv}, & \text{otherwise}.
\end{cases}
]

This is based on the provided SMP document, where SMP is defined as doubling single-node contributions while keeping double-node contributions unchanged. The document also states that this makes the first moment of the aggregated message time-invariant under the time-invariant feature-mean assumption. 

### Important implementation goal

Please implement SMP as a **drop-in graph preprocessing / message-passing wrapper** that can be applied to baseline GNN models.

The code should support at least:

1. **GCN**
2. **GraphSAGE**
3. **GAT**, if possible
4. Any generic PyG model that accepts `edge_index` and optionally `edge_weight`

The main idea is:

* Take the original graph.
* Use node timestamps.
* Compute edge weights according to SMP.
* Use these edge weights during message passing whenever the model supports weighted aggregation.
* For layers that do not directly support edge weights, provide an alternative implementation or explain the limitation.

### Inputs

Assume we have:

```python
data.x              # node features, shape [num_nodes, num_features]
data.y              # labels
data.edge_index     # shape [2, num_edges]
data.node_year      # or data.time, shape [num_nodes]
data.train_mask
data.val_mask
data.test_mask
```

The timestamp tensor may be named `data.node_year`, `data.time`, or another configurable key.

The graph may be directed or undirected. The SMP rule should be computed per directed edge (u \to v), where (u) is the source neighbor and (v) is the target node.

In PyG convention:

```python
src = edge_index[0]
dst = edge_index[1]
```

so the SMP rule should be:

```python
delta = abs(time[src] - time[dst])
radius = min(t_max - time[dst], time[dst] - t_min)
single = delta > radius
edge_weight = 2.0 if single else 1.0
```

### Required functions/classes

Please write the following:

#### 1. SMP edge-weight constructor

```python
def compute_smp_edge_weight(
    edge_index,
    node_time,
    t_min=None,
    t_max=None,
    normalize=True,
    dtype=torch.float32,
):
    """
    Compute SMP edge weights for each directed edge u -> v.

    Args:
        edge_index: LongTensor [2, num_edges]
        node_time: Tensor [num_nodes]
        t_min: optional scalar
        t_max: optional scalar
        normalize: if True, normalize weights for stable aggregation
        dtype: output dtype

    Returns:
        edge_weight: Tensor [num_edges]
    """
```

Details:

* If `t_min` or `t_max` is not given, infer from `node_time`.
* Raw SMP weights are 2 for single nodes and 1 otherwise.
* If `normalize=True`, implement a sensible normalization compatible with GCN-style aggregation.
* Also allow `normalize=False` for raw SMP weights.
* Explain clearly what normalization is used.

#### 2. SMP preprocessing wrapper

```python
def apply_smp_to_data(
    data,
    time_attr="node_year",
    edge_weight_attr="edge_weight",
    normalize=True,
):
    """
    Adds SMP edge weights to a PyG Data object.
    """
```

This should add:

```python
data.edge_weight = smp_edge_weight
```

or another configurable attribute.

#### 3. SMP-compatible GCN model

Write a baseline GCN model where each `GCNConv` receives `edge_weight`.

Example:

```python
class SMPGCN(torch.nn.Module):
    ...
    def forward(self, x, edge_index, edge_weight=None):
        ...
```

#### 4. SMP-compatible GraphSAGE model

PyG `SAGEConv` may not support `edge_weight` directly depending on version. Please handle this carefully.

Options:

* Implement a custom weighted GraphSAGE convolution, or
* Use a message-passing subclass that multiplies source messages by `edge_weight`.
* Make sure the aggregation is still mean-like, not just sum-like.
* The SMP denominator should match weighted averaging:

[
M_v
===

\frac{\sum_{u\in \mathcal{N}(v)} w_{uv} x_u}
{\sum_{u\in \mathcal{N}(v)} w_{uv}}.
]

So for GraphSAGE, implement weighted mean aggregation.

#### 5. Optional SMP-GAT

If implementing GAT with SMP is tricky, give one of these:

* A version where SMP weights multiply attention coefficients before normalization.
* Or a clear explanation of why naïvely applying SMP to GAT is not equivalent to the SMP averaging formula.

#### 6. Training loop

Provide a complete training and evaluation loop:

```python
def train(model, data, optimizer, criterion):
    ...

@torch.no_grad()
def evaluate(model, data):
    ...
```

Use masks:

```python
data.train_mask
data.val_mask
data.test_mask
```

Return accuracy.

#### 7. Example usage

Show example usage:

```python
data = apply_smp_to_data(data, time_attr="node_year", normalize=True)

model = SMPGCN(
    in_channels=data.num_features,
    hidden_channels=256,
    out_channels=num_classes,
    num_layers=2,
    dropout=0.5,
)

out = model(data.x, data.edge_index, data.edge_weight)
```

Also show how to switch between:

* Vanilla GCN
* SMP-GCN
* Vanilla GraphSAGE
* SMP-GraphSAGE

### Coding style requirements

Please make the code:

* Clean
* Modular
* Easy to copy into an existing PyTorch Geometric project
* Fully typed where reasonable
* GPU-compatible
* Robust to `node_time` being int, float, or year values
* Robust to missing `edge_weight`
* Well-commented but not over-commented

### Mathematical correctness requirements

Please be careful about edge direction.

For each edge (u \to v):

* (u) is the neighbor/source node.
* (v) is the target/destination node.
* The radius should be computed using the target timestamp (time(v)), not the source timestamp.

The SMP condition is:

[
|time(u)-time(v)|

>

\min(t_{\max}-time(v), time(v)-t_{\min}).
]

If this condition holds, assign weight 2. Otherwise assign weight 1.

For weighted mean aggregation, the message should be:

[
M_v
===

\frac{
\sum_{u\in \mathcal{N}(v)} w_{uv} x_u
}{
\sum_{u\in \mathcal{N}(v)} w_{uv}
}.
]

Do not accidentally use unnormalized weighted sums unless the model explicitly requires it.

### Deliverable

Please produce a single Python file containing:

1. SMP edge-weight computation
2. PyG data preprocessing
3. SMP-GCN
4. SMP-GraphSAGE with weighted mean aggregation
5. Optional SMP-GAT
6. Training/evaluation utilities
7. Example usage block under:

```python
if __name__ == "__main__":
    ...
```