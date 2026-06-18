# jam op coverage: StableHLO → MPSGraph

The op-mapping spec for **jam** (the mccl-jax compile side: StableHLO → MPSGraph).
jam walks the StableHLO MLIR module and emits one MPSGraph node per op — no custom IR,
MPSGraph handles fusion/scheduling. This table is the contract the C++ lowering implements.

Every row below was validated numerically on the Apple GPU vs jax-CPU (max abs err < 1e-4
for f32): **125/125 cases pass**, ~90 distinct StableHLO/CHLO ops plus real-model
compositions. The runnable reference that produced these (build-free Python+ObjC++ oracle)
is archived at `/tmp/mccl-jax-tests/jam-prototype/`; it is the oracle, not the product.

**Goal: total coverage.** The full op universe is **115 StableHLO + 49 CHLO = 164 ops**.
jam must give every one a *directly-linked* handler so that *any* StableHLO module compiles —
no op falls through. The coverage matrix at the bottom enumerates all 164 and their status.
Part A below is the validated lowering contract; the matrix is the completeness checklist.

## Op-mapping table (StableHLO → MPSGraph)

| StableHLO op | MPSGraph selector | Notes |
|---|---|---|
| `add` / `subtract` / `multiply` / `divide` | `additionWithPrimaryTensor:` / `subtractionWithPrimaryTensor:` / `multiplicationWithPrimaryTensor:` / `divisionWithPrimaryTensor:` | |
| `maximum` / `minimum` | `maximumWithPrimaryTensor:` / `minimumWithPrimaryTensor:` | relu = `maximum(x, 0)` |
| `power` | `powerWithPrimaryTensor:` | |
| `negate` | `negativeWithTensor:` | |
| `exponential` | `exponentWithTensor:` | |
| `log` | `logarithmWithTensor:` | |
| `tanh` | `tanhWithTensor:` | |
| `rsqrt` | `reciprocalSquareRootWithTensor:` | |
| `sqrt` | `squareRootWithTensor:` | |
| `abs` | `absoluteWithTensor:` | |
| `logistic` | `sigmoidWithTensor:` | |
| `floor` / `ceil` / `sine` / `cosine` / `tan` | `floorWithTensor:` / `ceilWithTensor:` / `sinWithTensor:` / `cosWithTensor:` / `tanWithTensor:` | |
| `sign` | `signWithTensor:` | |
| `round_nearest_even` / `round_nearest_afz` | `rintWithTensor:` / `roundWithTensor:` | half-to-even vs half-away-from-zero |
| `exponential_minus_one` (expm1) | `exp(x) − 1` | composed |
| `log_plus_one` (log1p) | `log(1 + x)` | composed |
| `cbrt` | `sign(x)·\|x\|^(1/3)` | composed (pow undefined for negative bases) |
| `chlo.erf` | `erfWithTensor:` | |
| `chlo.asin/acos/atan` | `asinWithTensor:` / `acosWithTensor:` / `atanWithTensor:` | JAX keeps inverse-trig in the CHLO dialect |
| `chlo.sinh/cosh/asinh/acosh/atanh` | `sinhWithTensor:` … `atanhWithTensor:` | |
| `atan2` | `atan2WithPrimaryTensor:` | |
| `remainder` | `moduloWithPrimaryTensor:` | fmod semantics (sign of dividend) |
| `and` / `or` / `xor` | `bitwiseAND/OR/XOR…` (int) or `logicalAND/OR/XOR…` (bool) | picked by operand dtype |
| `not` | `bitwiseNOTWithTensor:` (int) / `notWithTensor:` (bool) | |
| `shift_left` / `shift_right_arithmetic` | `bitwiseLeftShift…` / `bitwiseRightShift…` | |
| `shift_right_logical` | reinterpret→uint32, `bitwiseRightShift…`, reinterpret→int32 | |
| `popcnt` | `bitwisePopulationCountWithTensor:` | |
| `clamp` | `clampWithTensor:minValueTensor:maxValueTensor:` | |
| `is_finite` | `!(isInfinite ∥ isNaN)` | composed |
| `constant` | `constantWithScalar:shape:dataType:` (splat) / `constantWithData:shape:dataType:` (dense) | non-finite values (`±inf`, `nan`) need care |
| `dot_general` (general) | `matrixMultiplicationWithPrimaryTensor:secondaryTensor:` | single contracting dim; batched + non-canonical layouts handled by transposing each operand to `[batch…, free, contract]` / `[batch…, contract, free]` then MPSGraph leading-dim broadcast matmul |
| `convolution` (2D) | `convolution2DWithSourceTensor:weightsTensor:descriptor:` | any 2-spatial-dim layout → NHWC data / HWIO weights; explicit padding, strides, dilation, feature groups; result permuted back |
| `reduce` (add/max/min/mul/and/or) | `reductionSumWithTensor:axis:` etc. | reduce per-axis, then `reshapeTensor:` to drop kept dims (MPSGraph reductions keep dims); init operand ignored |
| `reduce` (variadic argmax/argmin) | `reductionArgMaximumWithTensor:axis:` / `reductionArgMinimumWithTensor:` | the 2-in/2-out reduce JAX emits is pattern-matched (GT⇒argmax, LT⇒argmin) |
| `reduce_window` (cumulative) | `cumulativeSumWithTensor:axis:` / `cumulativeProductWithTensor:` | full-extent window + `[N−1,0]`/`[0,N−1]` padding ⇒ forward/reverse cumsum/cumprod |
| `broadcast_in_dim` | `reshapeTensor:` + `broadcastTensor:toShape:` | reshape to output rank with 1s in non-mapped dims, then broadcast |
| `reshape` | `reshapeTensor:withShape:` | |
| `transpose` | `transposeTensor:permutation:` | |
| `slice` | `sliceTensor:starts:ends:strides:` | contiguous and strided |
| `dynamic_slice` (static idx) | `sliceTensor:starts:ends:strides:` | only when every start index is a compile-time constant; starts clamped in-bounds |
| `dynamic_update_slice` (static idx) | `sliceUpdateDataTensor:updateTensor:starts:ends:strides:` | constant starts only; writes the update over `[start, start+update_dim)`, clamped in-bounds |
| `reverse` | `reverseTensor:axes:` | |
| `pad` | `padTensor:withPaddingMode:leftPadding:rightPadding:constantValue:` | low/high padding; fill value traced to its splat constant |
| `iota` | `coordinateAlongAxis:withShape:` + `castTensor:` | |
| `bitcast_convert` | `reinterpretCastTensor:toType:` | same-width reinterpret |
| `gather` (single-axis) | `gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:` | canonical take/embedding-lookup (one collapsed slice dim, full slices elsewhere) |
| `scatter` (single-axis) | `scatterNDWithDataTensor:updatesTensor:indicesTensor:batchDimensions:mode:` | set/add/min/max/mul; one inserted dim; scatter axis moved to front when ≠ 0 |
| `sort` (single key) | `sortWithTensor:axis:descending:` | reads comparator direction (LT/LE⇒asc, GT/GE⇒desc) past JAX's NaN preamble |
| `chlo.top_k` | `topKWithSourceTensor:axis:k:` | returns (values, indices) |
| `concatenate` | `concatTensors:dimension:` | |
| `select` | `selectWithPredicateTensor:truePredicateTensor:falsePredicateTensor:` | |
| `convert` | `castTensor:toType:` | dtype cast |
| `compare` (GT/GE/LT/LE/EQ/NE) | `greaterThanWithPrimaryTensor:` etc. | yields a bool tensor |

**dtypes:** f32 (primary), f16, i32, bool all validated. i64 narrowed to i32 (Metal has no
native 64-bit int). Scalars (rank-0) modeled as shape `[1]` (MPSGraph has no rank-0 tensor).

**Validated compositions:** relu, softmax, log-softmax, logsumexp (numerically-stable);
GELU (exact + tanh), LayerNorm, attention scores (`einsum bik,bjk->bij → /√d → softmax`);
MLP forward (`matmul → bias add → relu → matmul → bias add → logsumexp`).

## Part B — complete coverage matrix (all 164 ops)

**Status (validated on derek's Apple M2 GPU vs jax-CPU):** **121/121 cases pass** across 24
families through the *real compiled jam* (`jam::Compile`→MPSGraph), **111 op handlers linked**
(force-loaded, none dead-stripped). Any unhandled op raises `jam: unhandled op '<name>'`; the
genuinely unmappable ops (fft, opaque custom_call, cholesky/triangular_solve via lapack
custom_call) raise a clear named compile error. Control flow (while/if/case/scan), rng,
batch_norm (inference+training), select_and_scatter, and single-rank collectives are all
validated. So: jam compiles cleanly, any StableHLO artifact compiles-or-errors-clearly.

Status legend: **L** linked + validated · **W** wired but unvalidated · **T** to-wire ·
**M** routed to mccl (collective split, not an MPSGraph node) · **S** structural, handled in
the module walk (not a graph node) · **N/A** deprecated / never emitted (JAX uses the modern
form) — still given a handler so "any HLO compiles." Target = full support; the genuinely
hard-to-map ops are flagged.

### StableHLO (115)

Elementwise unary — **L**: `abs cbrt ceil convert cosine exp expm1 floor is_finite log log1p
logistic neg not popcnt round round_nearest_even rsqrt sign sine sqrt tan tanh`.
**T**: `clz` (no MPSGraph popcount-of-leading-zeros prim → compose via bit ops), `real imag`
(complex), `reduce_precision` (bf16 round-trip; passthrough for f32).

Elementwise binary — **L**: `add and atan2 compare div max min mul or pow rem shift_left
shift_right_arithmetic shift_right_logical subtract xor`. **T**: `complex`.

Ternary — **L**: `clamp select`.

Reductions / windows — **L**: `reduce reduce_window`(cumulative + 2D max/sum pooling).
**T**: `select_and_scatter` (pool backward), `batch_norm_training batch_norm_inference
batch_norm_grad`.

Data movement — **L**: `broadcast_in_dim concatenate dynamic_slice dynamic_update_slice gather
iota pad reshape reverse scatter slice sort transpose bitcast_convert constant`. **T**: general
`gather`/`scatter` (multi-offset / batched index maps), interior/negative `pad`.

Linalg — **L**: `dot_general`(≤1 contracting dim), `convolution`(2D). **T**: `dot_general`
(contracting rank >1), `cholesky triangular_solve fft`, `custom_call` (per-call handlers; no
generic lowering).

Control flow — **T**: `while if case`. **S**: `return optimization_barrier after_all
create_token get_tuple_element tuple`.

Dynamism — **T**: `dynamic_broadcast_in_dim dynamic_reshape real_dynamic_slice dynamic_iota
dynamic_pad dynamic_gather dynamic_conv get_dimension_size set_dimension_size`.

RNG / quant — **T**: `rng rng_bit_generator uniform_quantize uniform_dequantize`.

Collectives — **M** (→ src/mccl/collective, split point): `all_reduce all_gather all_to_all
reduce_scatter collective_broadcast collective_permute partition_id replica_id`.

Host transfer — **T** (no MPSGraph equivalent; explicit unsupported unless a host path is added):
`infeed outfeed send recv`.

Deprecated / superseded — **N/A** (handler maps to the modern op so HLO still compiles):
`broadcast`→broadcast_in_dim, `dot`→dot_general, `einsum unary_einsum`→dot_general,
`torch_index_select`→gather, `cross_replica_sum`→all_reduce (M), `map` (inline subcomputation),
`composite` (inline decomposition).

### CHLO (49) — legalized upstream, not jam's direct input

**Architectural fact (verified):** jam's `Compile()` takes a StableHLO *portable artifact*
(`deserializePortableArtifact`), which is VHLO-encoded and carries **no CHLO** — CHLO is
legalized to StableHLO before the artifact is built. So jam receives CHLO ops only in their
decomposed StableHLO form (e.g. `chlo.erf` → a StableHLO polynomial), which the StableHLO
handlers above already cover (all such families validated through the real decomposed path).
The standing `chlo.*` handlers are therefore effectively dead for the production path; jam runs
`chlo-legalize-to-stablehlo` in its parse pipeline so any CHLO-bearing module still compiles.
The entries below are kept only for a hypothetical raw-MLIR (non-artifact) entry point.

Transcendental — **L** (via legalized StableHLO): `erf asin acos atan asinh acosh atanh sinh
cosh tan`. **T**: `erfc erf_inv digamma lgamma polygamma zeta bessel_i1e`.

Predicates / misc — **L**: `top_k`. **T**: `is_inf is_neg_inf is_pos_inf conj next_after
square`(=x·x) `constant_like ragged_dot`. **S/N/A**: `constant`.

Implicit-broadcast binary (`broadcast_add … broadcast_xor`, `broadcast_compare select complex
shift_* pow rem next_after polygamma zeta`) — **T**: decompose to `broadcast_in_dim` + the core
binary op (all **L**).

### Honest hard limits (map cleanly only with caveats, or not 1:1)
- `custom_call` — opaque; only known target strings can be handled, no generic lowering.
- `fft` / complex (`complex real imag conj`) — MPSGraph complex/FFT support is limited.
- `cholesky` / `triangular_solve` — no direct MPSGraph solver; needs composition or unsupported.
- host transfer (`infeed/outfeed/send/recv`) — outside the GPU graph.
- **bf16** unmapped; **f16 cumulative** rejected by MPSGraph (f32 only); **i64** narrowed to i32
  (Metal has no 64-bit int); rank-0 modeled as `[1]`.

Anything still **T** must end at **L** (validated) or, for a true hard limit, a *clear* compile
error naming the op — never a silent skip. That is the "any HLO compiles" bar.
