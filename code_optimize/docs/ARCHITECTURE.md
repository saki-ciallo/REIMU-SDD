# Architecture and data flow

## Top-level pipelines

### Backbone and pooling

```text
input_values [B, samples]
  -> frontend
hidden_states [B, sequence, hidden]
  -> backbone architecture
last_hidden_state [B, sequence, hidden]
  -> pooling
pooled_output [B, pooled_size]
  -> classifier
logits [B, num_labels]
```

The frontend may be linear frame projection, non-streaming SincNet, or one of
the supported Hugging Face SSL encoders. If the SSL native width differs from
the backbone width, the frontend owns the single explicit output projection.
Every adjacent dimension is checked while building `ADDConfig`.

### SSL and AASIST

```text
input_values -> SSL frontend -> AASIST graph encoder -> embedding -> classifier
```

AASIST returns its embedding and temporal, spectral, and master graph states.
The classifier remains separate and AASIST does not create logits internally.

## Backbone layer

`SequenceFeedForwardBlock` follows the FLA pre-norm block shape:

```text
residual = hidden
x = attn_norm(hidden)
x, attention, cache = mixer(x, ...)
hidden = residual + dropout(x)

residual = hidden
x = mlp_norm(hidden)
x, optional_aux_loss = GatedMLP_or_LatentMoE(x)
hidden = residual + dropout(x)
```

The mixer is an upstream FLA `Attention`, `Raven`, `GatedDeltaNet2`, or
`Mamba3`. Dense feed-forward uses upstream `GatedMLP`; routing and grouped
expert execution are isolated under `feedforward/` for LatentMoE.

Attention output slots are preserved across mixed stacks. Current FLA mixers do
not materialize attention maps, so `output_attentions=True` returns one `None`
placeholder per layer. Baseline cache uses FLA `Cache` and accumulates sequence
length across calls.

## Backbone schedules

Let `M(x)` denote one complete shared `BackboneStack` call and `N(x)` its
terminal post-norm.

### Baseline

```python
output = M(input)
```

Baseline evaluates one parameterized stack once. It does not add `N`.

### Looped

One shared module `F(x) = N(M(x))` is reused:

```python
state = input
for cycle in range(num_cycles):
    cycle_has_grad = cycle >= num_cycles - gradient_cycles
    with torch.set_grad_enabled(cycle_has_grad):
        if cycle == num_cycles - gradient_cycles:
            state = bridge_input_gradient(state, input)
        state = F(state)
return state
```

The parameters are shared across cycles. Earlier calls run without an autograd
graph; the first retained call reconnects the unchanged recurrent value to the
frontend through a zero-valued gradient bridge.

### HRM

H and L are distinct parameterized modules, each including its own post-norm:

```python
z_h = input
z_l = learned_low_level_initial_state
step = 0

for h_cycle in range(h_cycles):
    for _ in range(l_cycles):
        z_l = run_with_schedule(L, z_l, step)
        step += 1

    z_h = run_with_schedule(H, z_h + z_l, step)
    step += 1

    if h_cycle + 1 < h_cycles:
        z_l = z_h
```

The total recurrent-step count is
`h_cycles * (l_cycles + 1)`. The final `K` complete module calls retain
autograd history. H and L parameters remain trainable even when an earlier call
runs under `no_grad`; truncation controls activation history, not
`requires_grad`.

### Heterogeneous HRM

The recurrence is identical to HRM. H and L receive independent block
specifications and therefore may use different mixers or feed-forward types,
for example H=Attention and L=GDN2. Their depths must match so recurrent state
semantics remain explicit.

## Pooling and classifier

All pooling implementations share:

```text
pool-specific aggregation -> optional output activation
                          -> optional output norm
                          -> dropout
```

The default MHGAP path is:

```text
Linear -> split values/gates
       -> evidence = values * SiLU(gates)
       -> per-head temporal score from values
       -> softmax over sequence
       -> weighted sum of evidence
       -> concatenate heads
       -> optional output transform -> dropout
```

The classifier always exposes label-independent prediction logits. AMSoftmax
additionally exposes margin-adjusted `loss_logits` when labels are present.
This prevents evaluation metrics from using label-modified scores while keeping
cross-entropy or focal loss outside the model.

## Output contracts

All public output dataclasses use `torch.Tensor`, not dtype-specific aliases.
Mixed precision therefore remains accurately described:

- frontend output: `[B, S, H]`;
- backbone: hidden state, optional cache/hidden-state history/attention slots,
  optional MoE auxiliary loss;
- pooling: `[B, P]`;
- classifier and ADD model: `[B, C]` logits.
