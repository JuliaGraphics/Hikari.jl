# Shared surface BSDF storage experiment

This experiment extends the integrated small Diffuse/Sheen/Glossy storage to
Principled (including the canonical `Glass` preset), metallic Fresnel models,
refraction, translucency, Toon and subsurface scattering. It does not change
Hikari's default kernel selection. Native pbrt node models are not yet packed.

The larger record stores 124 logical bytes per weighted lobe, in local storage.
Setup reads the runtime program once; its argument type contains a capability
mask and capacity instead of the exact ordered node tuple. This differs from
merely packing the result of the old specialized setup.

In an environment using these Hikari, Mantle and Lava checkouts:

```julia
include(joinpath(pkgdir(Hikari), "benchmarks", "shared_surface", "load.jl"))
SurfaceProgramClassChecks.run()
SurfaceProgramCPUCheck.run()
SurfaceProgramKernelMWE.run()
SurfaceProgramGPUProbe.compare(Hikari.Principled(); resolution=96, samples=8)
```

Import Hikari before using `pkgdir(Hikari)`. The GPU tests require a working
display on backends that initialize GLFW. Do not load `load.jl` repeatedly in one
session because it defines experimental types. Run tests in a separate session
from interactive work. `compare` restores kernel selection in `finally`.

Validation so far: 25,192 differential CPU checks over 67 configurations; the
standalone generic Principled GPU evaluation agrees with the original typed CPU
reference. Earlier full-image tests of the larger record used specialized setup
and passed for Principled, glass, metals and subsurface. They do not demonstrate
reuse of generic setup. Full generic renderer compilation is still under
investigation after worker terminations. Runtime and memory acceptance remain
open; this experiment must not be enabled by default on this evidence.

The related plan and acceptance criteria are in Mantle's
`docs/material-compilation-reuse-plan.md`.
