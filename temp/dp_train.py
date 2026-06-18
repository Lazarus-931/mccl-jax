#!/usr/bin/env python3
"""Data-parallel MLP training across the mini cluster through mccl-jax. Batch sharded over the
mesh; gradients pmean'd (all_reduce) each step so all ranks keep identical (replicated) params.

Run per rank:  ~/jaxenv/bin/python dp_train.py <rank> <nproc> <coord_ip:port> [hidden] [steps]
"""
import os
import sys
import time

a = sys.argv[1:]
rank, nproc, coord = int(a[0]), int(a[1]), a[2]
H = int(a[3]) if len(a) > 3 else 2048
STEPS = int(a[4]) if len(a) > 4 else 1500

# NB: do NOT set JAX_PLATFORMS=metal before jax.distributed.initialize — at nproc>1 that makes
# jdist.initialize try the not-yet-registered metal backend and HANG the coordination barrier. Select
# the platform via jax.config.update AFTER initialize + register_plugin (the working order).
os.environ.setdefault("MCCL_BOOTSTRAP_IP", coord.split(":")[0])
os.environ.setdefault("MCCL_BOOTSTRAP_PORT", "53700")
os.environ.setdefault("MCCL_METAL_DIR", os.path.expanduser("~/mccl/src/device"))

import numpy as np
import jax
import jax.numpy as jnp
import jax.distributed as jdist
jdist.initialize(coordinator_address=coord, num_processes=nproc, process_id=rank)

import jax._src.xla_bridge as xb
xb.register_plugin("metal", priority=400,
                   library_path=os.path.expanduser("~/mccl-jax-build/build/libpjrt_metal.dylib"))
jax.config.update("jax_platforms", "metal")

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

D = 1024
mesh = Mesh(np.array(jax.devices()), ('i',))
ld = jax.local_devices()[0]

irng = np.random.default_rng(0)  # same seed on every rank -> identical replicated params
def w(shape, sc):
    return (irng.standard_normal(shape) * sc).astype(np.float32)
params0 = [w((D, H), (2/D)**0.5), np.zeros(H, np.float32),
           w((H, H), (2/H)**0.5), np.zeros(H, np.float32),
           w((H, D), (2/H)**0.5), np.zeros(D, np.float32)]
nparams = sum(int(p.size) for p in params0)

B_local = 64
brng = np.random.default_rng(1000 + rank)   # this rank's distinct batch shard (fixed)
xr = brng.standard_normal((B_local, D)).astype(np.float32)
yr = brng.standard_normal((B_local, D)).astype(np.float32)

rep, shd = NamedSharding(mesh, P()), NamedSharding(mesh, P('i'))
def g_rep(arr):
    return jax.make_array_from_single_device_arrays(arr.shape, rep, [jax.device_put(jnp.asarray(arr), ld)])
def g_shd(arr, gshape):
    return jax.make_array_from_single_device_arrays(gshape, shd, [jax.device_put(jnp.asarray(arr), ld)])

params = [g_rep(p) for p in params0]
m = [g_rep(np.zeros_like(p)) for p in params0]
v = [g_rep(np.zeros_like(p)) for p in params0]
gx, gy = g_shd(xr, (nproc * B_local, D)), g_shd(yr, (nproc * B_local, D))
def rep_scalar(x):  # replicated device scalar (a raw host scalar has no mesh sharding)
    return g_rep(np.asarray(x, np.float32))

def fwd(p, x):
    W1, b1, W2, b2, W3, b3 = p
    h = jax.nn.relu(x @ W1 + b1); h = jax.nn.relu(h @ W2 + b2); return h @ W3 + b3
def loss(p, x, y):
    return jnp.mean((fwd(p, x) - y) ** 2)
B1, B2, EPS = 0.9, 0.999, 1e-8

# Adam bias-correction factors are computed host-side (bc1=1-B1**t, bc2=1-B2**t) and passed in,
# rather than `1-B1**t` in-graph: a device-scalar `t` crossing a jam segment boundary mis-broadcasts
# (known jam gap). Same math, no in-graph 1-B1**t.
def body(p, m, v, x, y, bc1, bc2, lr):
    lval, g = jax.value_and_grad(loss)(p, x, y)
    g = [jax.lax.pmean(gi, 'i') for gi in g]           # data-parallel gradient average
    m = [B1*mi + (1-B1)*gi for mi, gi in zip(m, g)]
    v = [B2*vi + (1-B2)*gi*gi for vi, gi in zip(v, g)]
    p = [pi - lr*(mi/bc1)/(jnp.sqrt(vi/bc2) + EPS) for pi, mi, vi in zip(p, m, v)]
    return p, m, v, jax.lax.pmean(lval, 'i')           # global (pre-update) loss

def step_impl(p, m, v, x, y, bc1, bc2, lr):
    return shard_map(body, mesh, in_specs=(P(), P(), P(), P('i'), P('i'), P(), P(), P()),
                     out_specs=(P(), P(), P(), P()), check_rep=False)(p, m, v, x, y, bc1, bc2, lr)
# Pin output shardings to replicated so outputs are concretely sharded and re-feedable next step.
step = jax.jit(step_impl, out_shardings=rep)
def bc(s):
    return float(1 - B1**s), float(1 - B2**s)

b1c, b2c = bc(1)
params, m, v, l = step(params, m, v, gx, gy, rep_scalar(b1c), rep_scalar(b2c), rep_scalar(1e-3))
jax.block_until_ready(l)
def scalar(z): return float(np.asarray(z.addressable_shards[0].data).ravel()[0])
if rank == 0:
    print(f"params={nparams/1e6:.1f}M  global_batch={nproc*B_local}  ranks={nproc}  step 0 loss={scalar(l):.4f}")

t0 = time.time()
for s in range(1, STEPS + 1):
    lr = 1e-3 if s < int(STEPS*0.6) else 1e-3 * 0.5 ** ((s - int(STEPS*0.6)) // max(1, STEPS // 20))
    b1c, b2c = bc(s)
    params, m, v, l = step(params, m, v, gx, gy, rep_scalar(b1c), rep_scalar(b2c), rep_scalar(lr))
    if s % 500 == 0 or s == STEPS:
        jax.block_until_ready(l)
        if rank == 0:
            print(f"step {s:<5} loss={scalar(l):.4e}  ({s/(time.time()-t0):.1f} steps/s)")

csum = float(sum(np.asarray(p.addressable_shards[0].data).astype(np.float64).sum() for p in params))
print(f"[rank {rank}] FINAL loss={scalar(l):.4e}  param_checksum={csum:.6f}")
