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

# Re-export Raycore types and functions that Trace uses
import Raycore: AbstractRay, Ray, RayDifferentials, apply, check_direction, scale_differentials
import Raycore: Bounds2, Bounds3, area, surface_area, diagonal, maximum_extent, offset, is_valid, inclusive_sides, expand
import Raycore: distance, distance_squared, bounding_sphere
# Note: lerp is defined in spectrum.jl for Spectrum, Float32, and Point3f
import Raycore: Transformation, translate, scale, rotate, rotate_x, rotate_y, rotate_z, look_at, perspective
import Raycore: swaps_handedness, has_scale
import Raycore: Triangle, TriangleMesh
import Raycore: AccelPrimitive, BVH, TLAS, StaticTLAS, TraversableTLAS, TLASHandle, Instance, world_bound, closest_hit, any_hit, sync!

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
abstract type BxDF end
abstract type Integrator end
abstract type Medium end

# Default no-op close for integrators without cached state
Base.close(::Integrator) = nothing

const Radiance = UInt8(1)
const Importance = UInt8(2)

struct Reflect end
struct Transmit end

const DO_ASSERTS = false
macro real_assert(expr, msg="")
    if DO_ASSERTS
        esc(:(@assert $expr $msg))
    else
        return :()
    end
end

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
include("sampler/sampler.jl")
include("textures/mapping.jl")
include("textures/basic.jl")
include("textures/texture-ref.jl")
include("textures/environment_map.jl")
include("materials/uber-material.jl")
include("reflection/Reflection.jl")
include("materials/bsdf.jl")
include("materials/material.jl")
include("materials/volume.jl")
include("materials/emissive.jl")
include("materials/coated-diffuse.jl")
include("materials/mix-material.jl")
include("materials/thin-dielectric.jl")
include("materials/diffuse-transmission.jl")
include("materials/coated-conductor.jl")

# Spectral rendering support (for PhysicalWavefront)
# spectral.jl, piecewise-linear.jl, metal-spectra.jl included above (before textures)
include("spectral/color.jl")
include("spectral/uplift.jl")
include("materials/spectral-eval.jl")
# Sobol sampler (needs mix_bits from spectral-eval.jl)
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
include("lights/sun_sky.jl")
include("lights/ambient.jl")
include("lights/environment.jl")
include("lights/light-sampler.jl")
include("integrators/sampler.jl")
include("integrators/sppm.jl")
include("integrators/fast-wavefront.jl")
# Unified work queue for wavefront integrators
include("integrators/workqueue.jl")
# PhysicalWavefront spectral path tracer
include("integrators/physical-wavefront/workitems.jl")
include("integrators/physical-wavefront/material-dispatch.jl")
include("integrators/physical-wavefront/lights.jl")
include("integrators/physical-wavefront/camera.jl")
include("integrators/physical-wavefront/intersection.jl")
include("integrators/physical-wavefront/material-eval.jl")
include("integrators/physical-wavefront/film-update.jl")
# VolPath volumetric path tracer
include("integrators/volpath/media.jl")
include("integrators/volpath/nanovdb.jl")
include("integrators/volpath/medium-dispatch.jl")
include("integrators/volpath/workitems.jl")
include("integrators/volpath/volpath-state.jl")
include("integrators/volpath/delta-tracking.jl")
include("integrators/volpath/medium-scatter.jl")
include("integrators/volpath/intersection.jl")
include("integrators/volpath/surface-eval.jl")
include("integrators/volpath/multi-material-eval.jl")
include("integrators/volpath/volpath.jl")
include("kernel-abstractions.jl")
# Postprocessing pipeline
include("postprocess.jl")

# Denoising
include("denoise.jl")

# include("model_loader.jl")

end
