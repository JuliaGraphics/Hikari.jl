# Software Vulkan CI

Use Julia 1.12 or later. The explicit sources in this project select the
unregistered backend and KernelInterface branches required by Hikari. The
normal package dependencies alone do not describe this development stack.

From the Hikari checkout:

```sh
julia --project=.ci .ci/setup.jl
HIKARI_TEST_LAVAPIPE=true HIKARI_PBRT_SPP=256 HIKARI_SKIP_PBRT_HW=true \
    xvfb-run -a julia --project=.ci .ci/run_tests.jl 1
```

`setup.jl` makes `.ci` the one environment of every CI job: it adds Hikari's
dependencies and test target, the docs' and the pbrt comparison's dependencies,
installs Hikari's lazy artifacts and precompiles everything once. In CI a setup
job does this and saves the depot; the test, comparison and docs jobs restore it
with the resolved project, check that nothing is stale for their runner, and
never run Pkg. All jobs build for the same `JULIA_CPU_TARGET`, so a depot
precompiled on one runner loads on the others.

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

`run_tests.jl` runs `test/runtests.jl` in this environment rather than through
`Pkg.test`, whose sandbox is another environment and would precompile again. It
keeps Julia's default bounds checking (`auto`): forced bounds checks, the
`Pkg.test` default, inject device exception paths unsupported by the GPU stack.
All test files, including subsurface rendering, run under this policy.

CompilerCaching is pinned to the published revision providing `ObjCache`.
The Lavapipe device checks also require that persistent cache to be enabled,
so CI cannot silently exercise only the session cache.

CI also renders a small Principled scene in two separate Julia processes using
`cache_probe.jl`. The second process must retrieve every shader from the cache
with zero SPIR-V and LLVM optimization misses, and both images must be finite
and nonblack. Elapsed time is reported separately from these deterministic
cache assertions.
The cached render must also match the first process's image hash exactly.

The two probe processes share a dedicated `JULIA_OBJCACHE_PATH`. The general
object cache is bounded and shared by concurrent Julia processes; a large test
suite can evict freshly compiled probe shaders. Isolating this small working set
tests persistence without depending on unrelated cache traffic.

CI runs the shared test registry in three disjoint groups plus two pbrt
reference groups, each in a fresh process. This releases compiler and driver state
between groups; a complete single-process GPU run can exceed 10 GiB of resident
memory. `.ci/run_tests.jl 1` through `3` select the unit groups. `pbrt-materials`
and `pbrt-textures-other` divide the complete 256-spp reference gate; any
`pbrt-<category>[-<category>]` works, and `pbrt` selects all three categories
for local use. The aggregate `Tests` check requires every group to succeed.
`Pkg.test("Hikari")` still runs everything by default.
