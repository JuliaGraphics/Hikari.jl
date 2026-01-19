struct BXDFVector{S<:Spectrum}
    bxdf_1::UberBxDF{S}
    bxdf_2::UberBxDF{S}
    bxdf_3::UberBxDF{S}
    bxdf_4::UberBxDF{S}
    bxdf_5::UberBxDF{S}
    bxdf_6::UberBxDF{S}
    bxdf_7::UberBxDF{S}
    bxdf_8::UberBxDF{S}
    last::UInt8
end

Base.Base.@propagate_inbounds function Base.getindex(b::BXDFVector, i::Int)
    # i > b.last && error("Out of bounds $(i)")
    return getfield(b, i)
end

@propagate_inbounds function BXDFVector{S}(sbdfs::Vararg{UberBxDF{S}, N}) where {S<:Spectrum, N}
    missing_bxdf = ntuple(i -> UberBxDF{S}(), 8 - N)
    return BXDFVector{S}(sbdfs..., missing_bxdf..., UInt8(N))
end

struct BSDF{S}
    """
    Relative index of refraction over the boundary.
    For opaque surfaces it is not used and should be 1.
    """
    η::Float32
    """
    Geometric normal defined by the surface geometry.
    """
    ng::Normal3f
    """
    Shading normal defined by per-vertex normals and/or bump-mapping.
    These normals will generally define different hemispheres for
    integrating incident illumination to computer surface reflection.
    """
    ns::Normal3f
    """
    Component of the orhonormal coordinates system with the
    shading normal as one of the axes.
    """
    ss::Vec3f
    """
    Component of the orhonormal coordinates system with the
    shading normal as one of the axes.
    """
    ts::Vec3f

    """
    Individual BxDF components. Maximum allowed number of components is 8.
    """
    bxdfs::BXDFVector{S}
end

BSDF() = BSDF{RGBSpectrum}(NaN32, Normal3f(0f0), Normal3f(0f0), Vec3f(0f0), Vec3f(0f0), BXDFVector{RGBSpectrum}())


function BSDF(si::SurfaceInteraction, sbdfs::Vararg{UberBxDF{RGBSpectrum}, N}) where {N}
    BSDF(si, 1f0, sbdfs...)
end


function BSDF(si::SurfaceInteraction, η::Float32, sbdfs::Vararg{UberBxDF{RGBSpectrum},N}) where {N}
    ng = si.core.n
    ns = si.shading.n
    ss = normalize(si.shading.∂p∂u)
    ts = ns × ss
    bsdfs = BXDFVector{RGBSpectrum}(sbdfs...)
    BSDF{RGBSpectrum}(
        η, ng, ns, ss, ts, bsdfs,
    )
end


"""
Given the orthonormal vectors s, t, n in world space, the matrix `M`
that transforms vectors in world space to local reflection space is:
    sx, sy, sz
    tx, ty, tz
    nx, ny, nz

Since it is an orthonormal matrix, its inverse is its transpose.
"""
@propagate_inbounds function world_to_local(b::BSDF, v::Vec3f)
    Vec3f(v ⋅ b.ss, v ⋅ b.ts, v ⋅ b.ns)
end
# TODO benchmark
@propagate_inbounds function local_to_world(b::BSDF, v::Vec3f)
    Mat3f(b.ss..., b.ts..., b.ns...) * v
end

"""
Evaluate BSDF function given incident and outgoind directions.
"""
function (b::BSDF)(
        wo_world::Vec3f, wi_world::Vec3f, flags::UInt8 = BSDF_ALL,
    )::RGBSpectrum

    # Transform world-space direction vectors to local BSDF space.
    wo = world_to_local(b, wo_world)
    wo[3] ≈ 0f0 && return RGBSpectrum(0f0)
    wi = world_to_local(b, wi_world)
    # Determine whether to use BRDFs or BTDFs.
    reflect = ((wi_world ⋅ b.ng) * (wo_world ⋅ b.ng)) > 0

    output = RGBSpectrum(0f0)
    bxdfs = b.bxdfs
    bxdfs.last == 0 && return output
    Base.Cartesian.@nexprs 8 i -> begin
        # GPU-compatible: use early return instead of @assert
        i > bxdfs.last && return output
        bxdf = bxdfs[i]
        if ((bxdf & flags) && (
                (reflect && (bxdf.type & BSDF_REFLECTION != 0)) ||
                (!reflect && (bxdf.type & BSDF_TRANSMISSION != 0))
            ))
            output += bxdf(wo, wi)
        end
    end
    return output
end

# GPU-safe integer conversion helpers (avoid InexactError)
# Truncate toward zero (signed)
u_int32(x) = Base.unsafe_trunc(Int32, x)
u_int(x) = Base.unsafe_trunc(Int, x)
# Truncate toward zero (unsigned)
u_uint32(x) = Base.unsafe_trunc(UInt32, x)
u_uint64(x) = Base.unsafe_trunc(UInt64, x)
# Floor then truncate (for array indexing with potential negative intermediates)
floor_int32(x) = Base.unsafe_trunc(Int32, floor(x))
floor_int(x) = Base.unsafe_trunc(Int, floor(x))
# Round then truncate (for nearest-neighbor lookups)
round_int32(x) = Base.unsafe_trunc(Int32, round(x))
round_int(x) = Base.unsafe_trunc(Int, round(x))

"""
Compute incident ray direction for a given outgoing direction and
a given mode of light scattering corresponding
to perfect specular reflection or refraction.
"""
function sample_f(
        b::BSDF, wo_world::Vec3f, u::Point2f, type::UInt8,
    )::Tuple{Vec3f,RGBSpectrum,Float32,UInt8}

    # Choose which BxDF to sample.
    matching_components = num_components(b, type)
    matching_components == Int32(0) && return (
        Vec3f(0f0), RGBSpectrum(0f0), 0f0, BSDF_NONE,
    )
    component = min(
        max(Int32(1), u_int32(ceil(u[1] * matching_components))),
        matching_components,
    )
    # Get BxDF for chosen component.
    count = component
    component -= Int32(1)
    bxdf = UberBxDF{RGBSpectrum}()
    bxdfs = b.bxdfs
    Base.Cartesian.@nexprs 8 i -> begin
        if i <= bxdfs.last
            _bxdf = bxdfs[i]
            if _bxdf & type
                if count == 1
                    bxdf = _bxdf
                end
                count -= 1
            end
        end
    end
    @real_assert bxdf.active "n bxdfs $(b.n_bxdfs), component $component, count $count"
    # Remap BxDF sample u to [0, 1)^2.
    u_remapped = Point2f(
        min(u[1] * matching_components - component, 1f0), u[2],
    )
    # Sample chosen BxDF.
    wo = world_to_local(b, wo_world)
    wo[3] ≈ 0f0 && return (
        Vec3f(0f0), RGBSpectrum(0f0), 0f0, BSDF_NONE,
    )

    # TODO when to update sampled type
    sampled_type = bxdf.type

    wi, pdf, f, sampled_type_tmp = sample_f(bxdf, wo, u_remapped)
    sampled_type_tmp === BSDF_NONE || (sampled_type = sampled_type_tmp)

    pdf ≈ 0f0 && return (
        Vec3f(0f0), RGBSpectrum(0f0), 0f0, BSDF_NONE,
    )
    wi_world = local_to_world(b, wi)
    # Compute overall PDF with all matching BxDFs.
    if !(bxdf.type & BSDF_SPECULAR != Int32(0)) && matching_components > Int32(1)
        Base.Cartesian.@nexprs 8 i -> begin
            if i <= bxdfs.last && bxdfs[i] != bxdf && bxdfs[i] & type
                pdf += compute_pdf(bxdfs[i], wo, wi)
            end
        end
    end
    matching_components > Int32(1) && (pdf /= matching_components)
    # Compute value of BSDF for sampled direction.
    if !(bxdf.type & BSDF_SPECULAR != Int32(0))
        reflect = ((wi_world ⋅ b.ng) * (wo_world ⋅ b.ng)) > 0f0
        f = RGBSpectrum(0f0)
        Base.Cartesian.@nexprs 8 i -> begin
            if i <= bxdfs.last
                bxdf = bxdfs[i]
                if  ((bxdf & type) && (
                        (reflect && (bxdf.type & BSDF_REFLECTION != Int32(0))) ||
                        (!reflect && (bxdf.type & BSDF_TRANSMISSION != Int32(0)))
                    ))
                    f += bxdf(wo, wi)
                end
            end
        end
    end

    return wi_world, f, pdf, sampled_type
end

function compute_pdf(
        b::BSDF, wo_world::Vec3f, wi_world::Vec3f, flags::UInt8,
    )::Float32
    b.bxdfs.last == 0 && return 0f0
    wo = world_to_local(b, wo_world)
    wo[3] ≈ 0f0 && return 0f0
    wi = world_to_local(b, wi_world)
    pdf = 0f0
    matching_components = 0
    bxdfs = b.bxdfs
    Base.Cartesian.@nexprs 8 i -> begin
        bxdf = bxdfs[i]
        if i <= bxdfs.last && bxdf & flags
            matching_components += 1
            pdf += compute_pdf(bxdf, wo, wi)
        end
    end
    matching_components > 0 ? pdf / matching_components : 0f0
end

@propagate_inbounds function num_components(b::BSDF, flags::UInt8)::Int64
    num = Int32(0)
    bxdfs = b.bxdfs
    Base.Cartesian.@nexprs 8 i -> begin
        if i <= bxdfs.last && (bxdfs[i] & flags)
            num += Int32(1)
        end
    end
    return num
end
