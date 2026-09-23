module Hikari

using Base: @propagate_inbounds
import FileIO
using ImageCore
using ImageIO
using GeometryBasics
using LinearAlgebra
using StaticArrays
using ProgressMeter
using StructArrays
using Atomix
using KernelAbstractions
using Raycore
using Zlib_jll
using Adapt
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA
using GPUArraysCore: @allowscalar
# Mantle is a hard dependency: `hw-rt.jl` below imports it unconditionally.
import Mantle

# The device-side ray-tracing intrinsics `rt-pipeline.jl` calls. They are
# KernelInterface's, re-exported by Mantle, and they are IMPORTED BY NAME
# because a shader body writes them bare: an unimported name in a kernel is not
# a `MethodError` but a global variable access, which GPUCompiler reports as
# "unsupported" three frames from where the name is missing. This block was
# `import Lava: lava_rt_*` — the one line that made a renderer which reaches the
# GPU through Mantle for everything else name a backend, and that stopped Hikari
# loading on a machine with no Vulkan driver.
using Mantle: rt_launch_id_x, rt_hit_object_trace_ray, rt_reorder_thread,
    rt_hit_object_execute_shader, rt_ignore_intersection, rt_primitive_id,
    rt_instance_id, rt_instance_custom_index, rt_ray_tmax, rt_hit_bary_u,
    rt_hit_bary_v

# Unexported but API: RayMakie's pbrt bridge calls it, so it is not free to
# change shape. `public` says that without putting it in every `using`'s scope.
public attach_bump

# Re-export Raycore types and functions that Trace uses
import Raycore: AbstractRay, Ray, RayDifferentials, apply, check_direction, scale_differentials
import Raycore: Bounds2, Bounds3, area, surface_area, diagonal, maximum_extent, offset, is_valid, inclusive_sides, expand
import Raycore: distance, distance_squared, bounding_sphere
# Note: lerp is defined in spectrum.jl for Spectrum, Float32, and Point3f
import Raycore: Transformation, translate, scale, rotate, rotate_x, rotate_y, rotate_z, look_at, perspective
import Raycore: swaps_handedness, has_scale
import Raycore: Triangle
import Raycore: TLAS, StaticTLAS, TraversableTLAS, TLASHandle, world_bound, closest_hit, any_hit, sync!

# Legacy alias - InstanceHandle was renamed to TLASHandle
const InstanceHandle = TLASHandle
import Raycore: Normal3f, intersect, intersect_p
import Raycore: is_dir_negative, increase_hit, intersect_p!
import Raycore: to_gpu
import Raycore: sum_unrolled, reduce_unrolled, for_unrolled, map_unrolled, getindex_unrolled
import Raycore: SetKey, MultiTypeSet, StaticMultiTypeSet, with_index, is_invalid, n_slots

abstract type Spectrum end
abstract type Light end
abstract type Material end
abstract type Integrator end
abstract type Medium end

"""
    abstract type GeneratedGeometry <: Material

A material that carries its own geometry.

A plot normally hands a renderer triangles. Some things are not triangles: a
finite element is a polynomial patch, a subdivision surface is a limit, an
implicit surface is a root. What those share is that the TRIANGLES ARE A CHOICE
— how many, and how close — and the code that knows how to make them belongs to
whoever owns the data.

Subtype this and implement [`tessellate`](@ref). That is the whole interface:
`mesh!(scene, anything; material = yours)` then draws it, rasterised and path
traced, from the same triangles. The `anything` is what a backend that has never
heard of your material falls back to.

```julia
struct MyElements <: Hikari.GeneratedGeometry
    nodes::Matrix{Float64}
    tolerance::Float64
end

Hikari.tessellate(m::MyElements) = GeometryBasics.Mesh(positions, faces)
```

**If you can solve a ray against your surface**, add a `push!` method and the
tracer stops approximating — one box per primitive goes into the acceleration
structure and the hit is exact at any zoom, while the rasteriser goes on using
`tessellate`. Dispatch picks it; nothing branches on a flag:

```julia
Base.push!(scene::Hikari.Scene, ::Any, m::MyElements; transform) = ...
```

`FEMMaterial` in `fem.jl` is a worked example of both halves.
"""
abstract type GeneratedGeometry <: Material end

"""
    tessellate(material::GeneratedGeometry) -> GeometryBasics.Mesh

The triangles `material` wants drawn, at whatever fidelity it carries.

Called once per change of the material, by every renderer. The material is the
only argument on purpose: how finely to approximate is a property of the thing
being drawn, so it belongs beside the data rather than in a renderer's settings.
A tolerance knob is then a field of your material, and moving it is
`plot.material[] = rebuilt`.
"""
function tessellate end

"""
    shadingmaterial(m::GeneratedGeometry) -> Material

The BSDF a generated geometry's triangles are shaded with. `Diffuse()` unless
you say otherwise.

Separate from the geometry because a `GeneratedGeometry` IS a material, and
pushing its own triangles into a scene with itself as the material would be a
loop. This is where that loop ends.
"""
shadingmaterial(::GeneratedGeometry) = Diffuse()

# Default no-op close/clear for integrators without cached state
Base.close(::Integrator) = nothing
clear!(::Integrator) = nothing

const DO_ASSERTS = false
macro real_assert(expr, msg="")
    if DO_ASSERTS
        esc(:(@assert $expr $msg))
    else
        return :()
    end
end

# Where a render state's device memory comes from. First, because everything
# below that allocates on the device takes a `DeviceMemory` in its signature —
# the spectral tables, the light BVH, the Sobol matrices, the work queues.
include("device-memory.jl")

include("spectrum.jl")
# PiecewiseLinearSpectrum needs SampledSpectrum/Wavelengths from spectral.jl,
# and must be available before texture-ref.jl and uber-material.jl
include("spectral/spectral.jl")
include("spectral/piecewise-linear.jl")
include("spectral/metal-spectra.jl")

include("random.jl")
include("surface_interaction.jl")
include("materials/medium-interface.jl")
include("scene.jl")

include("filter.jl")
include("film.jl")

include("camera/camera.jl")
include("sampler/sampling.jl")
include("textures/mapping.jl")
include("textures/basic.jl")
include("textures/texture-ref.jl")
include("textures/tex-handle.jl")
include("textures/environment_map.jl")

# Spectral rendering support
include("spectral/color.jl")
include("spectral/uplift.jl")
include("spectral/sensor.jl")
include("spectral/sensor_data.jl")

# Materials: shared math first, then each material, then dispatch
include("materials/common.jl")
include("materials/material.jl")
include("materials/diffuse.jl")
include("materials/dielectric.jl")
include("materials/conductor.jl")
include("materials/coated-diffuse.jl")
include("materials/mix-material.jl")
include("materials/coated-conductor.jl")
include("materials/coated-diffuse-transmission.jl")
include("materials/diffuse-transmission.jl")
include("materials/emissive.jl")
include("materials/bump-mapped.jl")
include("materials/merge-color.jl")
include("materials/dispatch.jl")

# Sobol sampler (needs mix_bits from materials/common.jl)
include("sampler/sobol_matrices.jl")
include("sampler/sobol.jl")
# Stratified sampler (needs murmur_hash_64a from spectral-eval.jl, sobol functions from sobol.jl)
include("sampler/stratified.jl")
include("primitive.jl")
include("lights/emission.jl")
include("lights/light.jl")
include("lights/point.jl")
include("lights/spot.jl")
include("lights/directional.jl")
include("lights/sun.jl")
include("lights/hosek_wilkie_data.jl")
include("lights/sun_sky.jl")
include("lights/ambient.jl")
include("lights/environment.jl")
include("lights/diffuse-area.jl")
include("lights/light-bounds.jl")
include("lights/light-sampler.jl")
include("lights/bvh-light-sampler.jl")
# GB.Mesh push! API (needs materials, lights, and DiffuseAreaLight)
include("scene-mesh.jl")
# Unified work queue for wavefront integrators
include("integrators/workqueue.jl")
# Spectral light sampling (used by VolPath for direct lighting, environment evaluation)
include("lights/spectral-sampling.jl")
# VolPath volumetric path tracer
include("integrators/volpath/media.jl")
include("integrators/volpath/nanovdb.jl")
include("integrators/volpath/medium-dispatch.jl")
include("integrators/volpath/workitems.jl")
include("integrators/volpath/per_material_queues.jl")
include("integrators/volpath/volpath-state.jl")
include("integrators/volpath/delta-tracking.jl")
include("integrators/volpath/medium-scatter.jl")
include("integrators/volpath/intersection.jl")
# After `intersection.jl`: `fem.jl` adds methods to the `vp_compute_*` helpers
# defined there, which is how an FEM element becomes a surface the integrator
# can shade without a second code path.
include("fem.jl")
include("integrators/volpath/surface-eval.jl")
# The sample as Mantle graphs — needs every stage's kernel to exist
include("integrators/volpath/graph.jl")
include("integrators/volpath/volpath.jl")
# Hardware RT dispatch (uses Raycore's backend-agnostic HWTLAS/AdaptedAccel interface)
include("integrators/volpath/hw-rt.jl")
# RT pipeline (raygen+closesthit+miss via VkRayTracingPipelineKHR + SBT) variant
include("integrators/volpath/rt-pipeline.jl")
include("kernel-abstractions.jl")
# Postprocessing pipeline
include("postprocess.jl")

# Denoising
include("denoise.jl")

# PBRT file parser
include("pbrt/tokenizer.jl")
include("pbrt/parser.jl")
include("pbrt/scene_builder.jl")

# include("model_loader.jl")

# ── Precompile workload ─────────────────────────────────────────────────────
#
# Hikari had no workload at all, so the first scene built in a session paid to
# JIT the parser and the whole scene builder. Measured on a 128x128 pbrt scene:
# 12.8 s for the first `load_pbrt`, 0.04 s for the second, i.e. essentially all
# of it was first-call latency rather than work.
#
# DEVICE-FREE: `build_hikari_scene` defaults to a CPU backend, so this
# never touches the Vulkan driver — precompilation must not, and a device crash
# here has poisoned pkgimages before (see Lava's `__init__`).
#
# A CPU-backend build cannot specialise the Lava-array half of the builder, so
# this does not eliminate the cost, it removes the backend-independent part:
# parsing, material and texture construction, the spectral tables and the BVH.
# Measured: parse 0.8 s + CPU build 5.3 s up front cuts the subsequent Lava-backed
# build from 12.8 s to 9.3 s.
#
# `Mantle.@setup_workload`/`@compile_workload`: PrecompileTools' macros re-exported
# by Mantle (the latter also wrapping the frozen-kernel recording,
# a no-op here since a CPU build compiles no SPIR-V), which is why Hikari needs
# no direct PrecompileTools dependency.
const _PRECOMPILE_SCENE = """
Film "rgb" "integer xresolution" 16 "integer yresolution" 16
LookAt 0 -1.2 0.6   0 0 0.5   0 0 1
Camera "perspective" "float fov" 40
Integrator "volpath" "integer maxdepth" 3
WorldBegin
LightSource "point" "rgb I" [40 40 40] "point3 from" [2 -1.5 3]
Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
Shape "trianglemesh"
  "point3 P" [ -1 0 -1  1 0 -1  1 0 1  -1 0 1 ]
  "normal N" [ 0 -1 0  0 -1 0  0 -1 0  0 -1 0 ]
  "point2 uv" [ 0 0  1 0  1 1  0 1 ]
  "integer indices" [ 0 1 2  0 2 3 ]
"""

include("precompile_statements.jl")

Mantle.@setup_workload begin
    Mantle.@compile_workload "hikari_scene_1" begin
        pbrt = parse_pbrt_string(_PRECOMPILE_SCENE)
        # A CPU build specialises the scene builder on Array-backed types that a
        # Lava-only user never calls, so it is fair to ask whether it is dead
        # weight. Measured with the precompile statements below already in place:
        # dropping it costs 0.96 s (load_pbrt 3.24 -> 4.08 s, render 3.70 ->
        # 3.82 s) and saves 12.4 MB of package image. It stays, because the
        # backend-independent half it covers -- the parser, materials, textures,
        # spectral tables -- is on every path, and because the CPU backend is a
        # real target: it is the reference the pbrt comparisons render against.
        build_hikari_scene(pbrt; backend = KernelAbstractions.CPU(), samples = 1,
                           max_depth = nothing, hw_accel = false)
        # The CPU build cannot reach anything parameterised on `LavaBackend`,
        # which is where the rest of the time was: ~29 s of Julia inference per
        # session, 22 s of it in ten signatures. `precompile` infers those
        # without running them, so this stays device-free.
        _precompile_statements()
    end
end

end
