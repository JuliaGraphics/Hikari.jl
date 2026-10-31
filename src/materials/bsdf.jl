"""
Maximum allowed number of BxDF components in BSDF.
"""
const MAX_BxDF = UInt8(8)

mutable struct BSDF
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
    Current number of BxDFs (≤ 8).
    """
    n_bxdfs::UInt8
    """
    Individual BxDF components. Maximum allowed number of components is 8.
    """
    bxdfs::Vector{B} where B<:BxDF

    function BSDF(si::SurfaceInteraction, η::Float32 = 1f0)
        ng = si.core.n
        ns = si.shading.n
        ss = normalize(si.shading.∂p∂u)
        ts = ns × ss
        new(
            η, ng, ns, ss, ts, UInt8(0),
            Vector{B where B<:BxDF}(undef, MAX_BxDF),
        )
    end
end

function add!(b::BSDF, x::B) where B<:BxDF
    @real_assert b.n_bxdfs < MAX_BxDF
    b.n_bxdfs += 1
    b.bxdfs[b.n_bxdfs] = x
end

"""
Given the orthonormal vectors s, t, n in world space, the matrix `M`
that transforms vectors in world space to local reflection space is:
    sx, sy, sz
    tx, ty, tz
    nx, ny, nz

Since it is an orthonormal matrix, its inverse is its transpose.
"""
@inline function world_to_local(b::BSDF, v::Vec3f)
    Vec3f(v ⋅ b.ss, v ⋅ b.ts, v ⋅ b.ns)
end
# TODO benchmark
@inline function local_to_world(b::BSDF, v::Vec3f)
    Mat3f0(b.ss..., b.ts..., b.ns...) * v
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
    for i in 1:b.n_bxdfs
        bxdf = b.bxdfs[i]
        if ((bxdf & flags) && (
            (reflect && (bxdf.type & BSDF_REFLECTION != 0)) ||
            (!reflect && (bxdf.type & BSDF_TRANSMISSION != 0))
        ))
            output += bxdf(wo, wi)
        end
    end
    output
end

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
    matching_components == 0 && return (
        Vec3f(0f0), RGBSpectrum(0f0), 0f0, BSDF_NONE,
    )
    component = min(
        max(1, Int64(ceil(u[1] * matching_components))),
        matching_components,
    )
    # Get BxDF for chosen component.
    count = component
    component -= 1
    bxdf = nothing
    for i in 1:b.n_bxdfs
        if b.bxdfs[i] & type
            count == 1 && (bxdf = b.bxdfs[i]; break)
            count -= 1
        end
    end
    @real_assert bxdf ≢ nothing "n bxdfs $(b.n_bxdfs), component $component, count $count"
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
    xx = sample_f(bxdf, wo, u_remapped)
    wi, pdf, f, sampled_type_tmp = xx
    sampled_type_tmp ≢ nothing && (sampled_type = sampled_type_tmp)

    pdf ≈ 0f0 && return (
        Vec3f(0f0), RGBSpectrum(0f0), 0f0, BSDF_NONE,
    )
    wi_world = local_to_world(b, wi)
    # Compute overall PDF with all matching BxDFs.
    if !(bxdf.type & BSDF_SPECULAR != 0) && matching_components > 1
        for i in 1:b.n_bxdfs
            if b.bxdfs[i] != bxdf && b.bxdfs[i] & type
                pdf += compute_pdf(b.bxdfs[i], wo, wi)
            end
        end
    end
    matching_components > 1 && (pdf /= matching_components)
    # Compute value of BSDF for sampled direction.
    if !(bxdf.type & BSDF_SPECULAR != 0)
        reflect = ((wi_world ⋅ b.ng) * (wo_world ⋅ b.ng)) > 0
        f = RGBSpectrum(0f0)
        for i in 1:b.n_bxdfs
            bxdf = b.bxdfs[i]
            if ((bxdf & type) && (
                (reflect && (bxdf.type & BSDF_REFLECTION != 0)) ||
                (!reflect && (bxdf.type & BSDF_TRANSMISSION != 0))
            ))
                f += bxdf(wo, wi)
            end
        end
    end

    wi_world, f, pdf, sampled_type
end

function compute_pdf(
    b::BSDF, wo_world::Vec3f, wi_world::Vec3f, flags::UInt8,
)::Float32
    b.n_bxdfs == 0 && return 0f0
    wo = world_to_local(b, wo_world)
    wo[3] ≈ 0f0 && return 0f0
    wi = world_to_local(b, wi_world)
    pdf = 0f0
    matching_components = 0
    for i in 1:b.n_bxdfs
        if b.bxdfs[i] & flags
            matching_components += 1
            pdf += compute_pdf(b.bxdfs[i], wo, wi)
        end
    end
    matching_components > 0 ? pdf / matching_components : 0f0
end

function num_components(b::BSDF, flags::UInt8)::Int64
    num = 0
    for i in 1:b.n_bxdfs
        (b.bxdfs[i] & flags) && (num += 1)
    end
    num
end
