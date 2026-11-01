# Software Vulkan CI

Use Julia 1.12 or later. The explicit sources in this project select the
unregistered backend and KernelInterface branches required by Hikari. The
normal package dependencies alone do not describe this development stack.

From the Hikari checkout:

```sh
julia --project=.ci .ci/setup.jl
HIKARI_TEST_LAVAPIPE=true HIKARI_PBRT_SPP=256 HIKARI_SKIP_PBRT_HW=true \
    xvfb-run -a julia --project=.ci -e 'using Pkg; Pkg.test("Hikari"; allow_reresolve=false, julia_args=`--check-bounds=auto`)'
```

`Lavapipe_jll` supplies the Vulkan driver. The test entry point loads it before
Hikari, restricts the Vulkan loader to its ICD, and asserts that the selected
device is llvmpipe. Xvfb provides the display used by GLFW; installing a system
Mesa Vulkan package is unnecessary. All ordinary tests run, including the
material differential, Cycles reference, graph update and shader reuse tests.
The full software pbrt reference suite retains its existing 256-sample gate.

The CI project has no committed Manifest. After updating this project's sources
in an existing checkout, regenerate its ignored Manifest or use `Pkg.update()`
to refresh the selected branch revisions. Restart persistent Julia sessions
after resolving dependencies before collecting compilation or runtime results.

On a hardware worker, leave `HIKARI_TEST_LAVAPIPE` unset. Hardware validation is
separate from the deterministic software-driver CI job.

GPU tests use ``Pkg.test("Hikari"; julia_args = `--check-bounds=auto`)``.
Forced bounds checks inject device exception paths unsupported by the GPU stack;
the normal policy retains checks where Julia cannot prove them unnecessary.
All test files, including subsurface rendering, run under this policy.

CompilerCaching is pinned to the published revision providing `ObjCache`.
The Lavapipe device checks also require that persistent cache to be enabled,
so CI cannot silently exercise only the session cache.

CI also renders a small Principled scene in two separate Julia processes using
`cache_probe.jl`. The second process must retrieve every shader from the cache
with zero SPIR-V and LLVM optimization misses, and both images must be finite
and nonblack. Elapsed time is reported separately from these deterministic
cache assertions.
