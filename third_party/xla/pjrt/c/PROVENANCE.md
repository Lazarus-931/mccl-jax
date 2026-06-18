# Vendored PJRT C API header

`pjrt_c_api.h` is vendored unmodified from OpenXLA. It is the **contract** the plugin
implements — it declares `GetPjrtApi`, the `PJRT_Api` function-pointer table, and every
`*_Args` struct. We depend only on this header, never on the XLA libraries.

| | |
|---|---|
| Source | https://github.com/openxla/xla — `xla/pjrt/c/pjrt_c_api.h` |
| Commit | `76da730179313b3bebad6dea6861768421b7358c` |
| Why | The XLA commit pinned by **jax/jaxlib 0.4.35**, so the PJRT API version and StableHLO contract match the jaxlib we target. |
| PJRT API version | major **0**, minor **55** |

jaxlib hard-errors on a PJRT API major mismatch, so `GetPjrtApi` reports `{0, 55}`. To target a
newer jaxlib: re-vendor from the XLA commit that jaxlib pins (see jax `third_party/xla/workspace.bzl`)
and update the version in `src/pjrt/api/pjrt_c_api.cc`.
