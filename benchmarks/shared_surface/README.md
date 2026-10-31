# Shared BSDF validation and optimization experiments

The 108-byte counted tuple storage with inline evaluation is integrated in
`src/materials/bsdf-storage.jl` and `bsdf-program.jl`. Normal Cycles rendering uses
it for individual nodes and mixtures. The former separate Diffuse/Sheen/Glossy
implementation has been removed. `Glass(...)` is a Principled preset. The native
pbrt component/layered adapters still specialize their schemas.

`load.jl` uses the production implementation. `renderer.jl` temporarily records
the original typed scattering formulas only for the differential reference.
`storage.jl`, `program.jl` and `counted_storage.jl` below are historical snapshots
of the experiments; do not include them into the integrated package. Alternative
layout/function-boundary experiments require isolated sessions and are not
renderer configuration options. The remainder records historical measurements. The active implementation is
the source under `src/materials/`; experiments below must not overwrite it in
an ordinary package session.

Current integration evidence (2026-10-31): 28,181 fresh-load checks pass for
storage, direct evaluation, mixed native graphs and all 73 differential model
configurations. All 17 Cycles reference batches pass (321 assertions), and
all 93 current nested RayMakie material/image update checks pass. Four different
Principled/Glass/Sheen/Glossy graphs reuse one native pipeline (28 checks), with
pixel-identical reference images. Subsequent unseen graphs take 0.274–0.279 s
including host setup and first rendering after that class is warm.
GGX/Beckmann and Fresnel variants now share microfacet geometry. This fixes the
isolated Glass/Beckmann native compile: a 1,921,576-byte shader compiled under a
4-GiB data-segment limit and rendered through the same production path. The
earlier per-distribution version could exceed 20 GB RSS during native compilation.
The compiler's internal cause is not established by shader size alone.

Final production measurements on NVIDIA RTX 4000 Ada, software tracing,
512² pixels, eight samples, five warmups and 51 alternating frame pairs:

| Material | Typed reference | Shared storage | Difference |
| --- | ---: | ---: | ---: |
| Layered Principled | 26.367 ms | 27.414 ms | +3.97% |
| Diffuse | 21.662 ms | 21.911 ms | +1.15% |
| Sheen | 21.853 ms | 22.248 ms | +1.81% |
| Glossy | 22.565 ms | 22.881 ms | +1.40% |
| Glass | 36.964 ms | 37.314 ms | +0.95% |
| Metallic | 22.890 ms | 24.322 ms | +6.25% |

These are full-frame medians, not isolated BSDF instruction timings. All six
image comparisons pass; maximum component error is 3.83e-6 (Glass). The roughly
4% accepted Principled result must not be generalized to every model.
The current driver does not expose register/scratch statistics through this
measurement, so these results do not establish a spill/register explanation.
The same `SurfaceProgramGPUProbe.compare` call below reproduces the paired
measurement with `resolution=512, samples=8, timing_samples=51`.

Reducing capacity from four records to one, with the same interpreter and
evaluation formulas, measured +3.54% for Diffuse, +3.62% for Sheen, +3.44% for
Glossy, and +8.56% for Metallic in the same paired protocol. Images pass.
The host now selects this one-slot class for one-lobe graphs; compositions use
the existing power-of-two classes starting at four. This retains one rendering
implementation. Moving from one to two lobes can select a new capacity class;
different one-lobe Diffuse/Sheen/Glossy nodes still share their class. The fresh
73-configuration CPU/model/canonical suite passes 28,377 checks. A fresh GPU run
passes all 340 reference/image/update assertions after that capacity change.
The final production repeat above measured +1.15–1.81% for Diffuse/Sheen/Glossy
and +6.25% for Metallic. Diffuse/Sheen/Glossy reuse the exact same native
pipeline, checked by native identity. Run-to-run differences do not establish
the native register/local-memory mechanism; Metallic retains a measured cost.

The current larger record stores 108 logical bytes per weighted lobe, in local storage
(the first version used 124). The smaller layout restores Principled's known unit
Schlick reflection tint instead of storing it; it retains coloured glass transmission.
Setup reads the runtime program once; its argument type contains a capability
mask and capacity instead of the exact ordered node tuple. This differs from
merely packing the result of the old specialized setup.

In an environment using these Hikari, Mantle and Lava checkouts:

```julia
include(joinpath(pkgdir(Hikari), "benchmarks", "shared_surface", "load.jl"))
SurfaceProgramClassChecks.run()
SurfaceProgramCPUCheck.run()
SurfaceProgramKernelMWE.run()
SurfaceProgramGPUProbe.compare(
    Hikari.Principled(metallic=0.35, coat_weight=0.4, sheen_weight=0.25, roughness=0.3);
    resolution=96, samples=8,
)
```

Import Hikari before using `pkgdir(Hikari)`. The GPU tests require a working
display on backends that initialize GLFW. Load the validation modules once per session. Run tests in a separate session
from interactive work. `compare` restores kernel selection in `finally`.

Validation so far: 25,192 differential CPU checks over 67 configurations; the
standalone generic Principled GPU evaluation agrees with the original typed CPU
reference. Earlier full-image tests of the larger record used specialized setup
and passed for Principled, glass, metals and subsurface. They do not demonstrate
reuse of generic setup. The full generic layered Principled renderer now executes
and agrees with the typed renderer (maximum absolute pixel difference 1.42e-6 in
the 512² test).
The 124-byte looped Principled shader is 1.45 MB versus the typed shader's 0.85 MB;
both report 255 registers. Paired 512², 8-sample frame measurements still show
roughly 9–11% overhead. This was an earlier candidate; its measurements do not describe the integrated variant.
The 108-byte counted variant reduced that measurement to 5.5% overhead. A
less layered Principled case exposed a reproducible full-render device loss;
broader native pipeline reuse and runtime acceptance are still open.

`counted_storage.jl` is a historical prototype for an isolated checkout.
It must not be included into the integrated package. It skips lifting zero-weight terms and carries the live count
alongside bounded storage. It preserves dispersion metadata from inactive
transmission terms. It still requires full-image and runtime acceptance.

`counts.jl`, included into Main using the production storage, compares all stored GPU
fields and sampling/evaluation with CPU references. `shared_reflection.jl`,
included into Hikari, is a separate optional experiment sharing microfacet
geometry between Fresnel models; 8,496 CPU comparisons passed, and its geometry-sharing idea is now incorporated in production with additional
GGX/Beckmann sharing. Do not load the old snapshot over that implementation.
Neither file is loaded by `load.jl`.

The related plan and acceptance criteria are in Mantle's
`docs/material-compilation-reuse-plan.md`.

Latest follow-up results (2026-10-31): inlining the shared evaluator avoids the
sparse-graph GPU fault and passes standalone setup/evaluation/sampling. Four
Principled/Glass/Sheen/Glossy graph combinations reuse the same native BSDF
pipeline (28 checks, maximum pixel error 5.96e-8). Sparse Principled also renders
pixel-identically with hardware ray tracing. Paired software frame measurements
still show 4–5% overhead. `inline_evaluation.jl` and
`inline_shared_reflection.jl` reproduce these optional variants; the latter
requires `shared_reflection.jl` to have been loaded first. These files do not fix
the underlying non-inlined function-boundary failure.

`mutable_storage.jl` is a rejected alternative: it rendered correctly but took
680.00 ms versus 31.57 ms for the typed reference at 512²/8 samples (51 paired
frames). Its mutable BSDF also fails the immutable storage contract. It remains
only as a reproducible diagnostic, not an integration candidate.

`deferred_padding.jl` tests a smaller change to counted tuple storage: every
live slot is written during setup, then only unused slots are initialized before
constructing the tuple. It passes 5,376 CPU checks and complete-render checks,
but measured 29.96 ms versus 28.01 ms typed (+6.95%), so it is not an improvement
over the prior inline tuple candidate.

`native_compile.c` is a compile-only Vulkan diagnostic for captured software-
tracing compute shaders. It requires Vulkan 1.3 headers and a Vulkan loader;
no GPU dispatch is submitted. Always bound its native compiler, for example:

```sh
prlimit --data=4294967296 timeout --kill-after=5s 60s ./native_compile RTX captured.spv
```

The independent compile checks compiler resource use only. Render correctness
still needs the Hikari image/reference tests. Hardware-ray-query shaders require
a matching device feature/layout setup and are outside this small probe's scope.
