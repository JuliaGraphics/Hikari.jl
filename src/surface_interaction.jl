struct Interaction
    """
    Intersection point in world coordinates.
    """
    p::Point3f
    """
    Time of intersection.
    """
    time::Float32
    """
    Negative direction of ray (for ray-shape interactions)
    in world coordinates.
    """
    wo::Vec3f
    """
    Surface normal at the point in world coordinates.
    """
    n::Normal3f
end

Interaction() = Interaction(Point3f(Inf), 0f0, Vec3f(0f0), Normal3f(0f0))

struct ShadingInteraction
    n::Normal3f
    ∂p∂u::Vec3f
    ∂p∂v::Vec3f
    ∂n∂u::Normal3f
    ∂n∂v::Normal3f
end

struct SurfaceInteraction
    core::Interaction
    shading::ShadingInteraction
    uv::Point2f

    ∂p∂u::Vec3f
    ∂p∂v::Vec3f
    ∂n∂u::Normal3f
    ∂n∂v::Normal3f

    ∂u∂x::Float32
    ∂u∂y::Float32
    ∂v∂x::Float32
    ∂v∂y::Float32
    ∂p∂x::Vec3f
    ∂p∂y::Vec3f

    SurfaceInteraction() = new()

    function SurfaceInteraction(
            core::Interaction, shading::ShadingInteraction, uv,
            ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
            ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y,
            ∂p∂x, ∂p∂y,
        )
        new(
            core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
            ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂p∂x, ∂p∂y
        )
    end
end

@propagate_inbounds function SurfaceInteraction(
        p::Point3f, time::Float32, wo::Vec3f, uv::Point2f,
        ∂p∂u::Vec3f, ∂p∂v::Vec3f, ∂n∂u::Normal3f, ∂n∂v::Normal3f, reverse_normal::Bool
    )

    n = normalize((∂p∂u × ∂p∂v))

    if reverse_normal
        n *= -1
    end

    core = Interaction(p, time, wo, n)
    shading = ShadingInteraction(n, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v)
    return SurfaceInteraction(
        core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
        0f0, 0f0, 0f0, 0f0, Vec3f(0f0), Vec3f(0f0)
    )
end

@propagate_inbounds function SurfaceInteraction(
        normal, hitpoint::Point3f, time::Float32, wo::Vec3f, uv::Point2f,
        ∂p∂u::Vec3f, ∂p∂v::Vec3f, ∂n∂u::Normal3f, ∂n∂v::Normal3f
    )
    core = Interaction(hitpoint, time, wo, normal)
    shading = ShadingInteraction(normal, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v)
    return SurfaceInteraction(
        core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
        0.0f0, 0.0f0, 0.0f0, 0.0f0, Vec3f(0.0f0), Vec3f(0.0f0)
    )
end

@propagate_inbounds function set_shading_geometry(
        si::SurfaceInteraction, tangent::Vec3f, bitangent::Vec3f,
        ∂n∂u::Normal3f, ∂n∂v::Normal3f, orientation_is_authoritative::Bool,
    )
    shading_n = normalize(tangent × bitangent)
    core_n = si.core.n
    if orientation_is_authoritative
        core_n = face_forward(si.core.n, si.shading.n)
    else
        shading_n = face_forward(si.shading.n, si.core.n)
    end

    shading = ShadingInteraction(shading_n, tangent, bitangent, ∂n∂u, ∂n∂v)
    core = Interaction(si.core.p, si.core.time, si.core.wo, core_n)
    return SurfaceInteraction(si; shading=shading, core=core)
end

is_surface_interaction(i::Interaction) = i.n != Normal3f(0)

@propagate_inbounds function SurfaceInteraction(
        si::SurfaceInteraction;
        core=si.core , shading=si.shading, uv=si.uv, ∂p∂u=si.∂p∂u, ∂p∂v=si.∂p∂v,
        ∂n∂u=si.∂n∂u, ∂n∂v=si.∂n∂v, ∂u∂x=si.∂u∂x, ∂u∂y=si.∂u∂y,
        ∂v∂x=si.∂v∂x, ∂v∂y=si.∂v∂y, ∂p∂x=si.∂p∂x, ∂p∂y=si.∂p∂y
    )
    SurfaceInteraction(
        core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂p∂x, ∂p∂y
    )
end

"""
Compute partial derivatives needed for computing sampling rates
for things like texture antialiasing.
"""
@propagate_inbounds function compute_differentials(si::SurfaceInteraction, ray::RayDifferentials)

    if !ray.has_differentials
        return SurfaceInteraction(si;
            ∂u∂x=0.0f0, ∂v∂x=0.0f0, ∂u∂y=0.0f0, ∂v∂y=0f0, ∂p∂x=Vec3f(0.0f0), ∂p∂y=Vec3f(0.0f0)
        )
    end

    # Estimate screen change in p and (u, v).
    # Compute auxiliary intersection points with plane.

    d = -(si.core.n ⋅ si.core.p)
    tx = (-(si.core.n ⋅ ray.rx_origin) - d) / (si.core.n ⋅ ray.rx_direction)
    ty = (-(si.core.n ⋅ ray.ry_origin) - d) / (si.core.n ⋅ ray.ry_direction)
    px = ray.rx_origin + tx * ray.rx_direction
    py = ray.ry_origin + ty * ray.ry_direction

    ∂p∂x = px - si.core.p
    ∂p∂y = py - si.core.p
    # Compute (u, v) offsets at auxiliary points.
    # Choose two dimensions for ray offset computation.
    n = abs.(si.core.n)
    if n[1] > n[2] && n[1] > n[3]
        dim = Point2(2, 3)
    elseif n[2] > n[3]
        dim = Point2(1, 3)
    else
        dim = Point2(1, 2)
    end
    # Initialization for offset computation.
    a = Mat2f(dim[1], dim[1], dim[2], dim[2])
    bx = Point2f(px[dim[1]] - si.core.p[dim[1]], px[dim[2]] - si.core.p[dim[2]])
    by = Point2f(py[dim[1]] - si.core.p[dim[1]], py[dim[2]] - si.core.p[dim[2]])
    sx = a \ bx
    sy = a \ by

    ∂u∂x, ∂v∂x = any(isnan.(sx)) ? (0f0, 0f0) : sx
    ∂u∂y, ∂v∂y = any(isnan.(sy)) ? (0f0, 0f0) : sy
    return SurfaceInteraction(si; ∂u∂x, ∂v∂x, ∂u∂y, ∂v∂y, ∂p∂x, ∂p∂y)
end

"""
If an intersection was found, it is necessary to determine, how
the surface's material scatters light.
`compute_scattering!` method evaluates texture functions to determine
surface properties and then initializing a representation of the BSDF
at the point.
"""
@propagate_inbounds function compute_scattering!(
        primitive, si::SurfaceInteraction, ray::RayDifferentials,
        allow_multiple_lobes::Bool = false, transport = Radiance,
    )
    si = compute_differentials(si, ray)
    return si, primitive(si, allow_multiple_lobes, transport)
end

@propagate_inbounds function le(::SurfaceInteraction, ::Vec3f)::RGBSpectrum
    # TODO right now return 0, since there is no area lights implemented.
    RGBSpectrum(0f0)
end

@propagate_inbounds function apply(t::Transformation, si::Interaction)
    return Interaction(
        t(si.p),
        si.time,
        normalize(t(si.wo)),
        normalize(t(si.n)),
    )
end

@propagate_inbounds function apply(t::Transformation, si::ShadingInteraction)
    n = normalize(t(si.n))
    ∂p∂u = t(si.∂p∂u)
    ∂p∂v = t(si.∂p∂v)
    ∂n∂u = t(si.∂n∂u)
    ∂n∂v = t(si.∂n∂v)
    return ShadingInteraction(n, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v)
end

@propagate_inbounds function apply(t::Transformation, si::SurfaceInteraction)
    # TODO compute shading normal separately
    core = apply(t, si.core)
    shading = apply(t, si.shading)
    ∂p∂u = t(si.∂p∂u)
    ∂p∂v = t(si.∂p∂v)
    ∂n∂u = t(si.∂n∂u)
    ∂n∂v = t(si.∂n∂v)
    ∂p∂x = t(si.∂p∂x)
    ∂p∂y = t(si.∂p∂y)
    return SurfaceInteraction(
        core, shading, si.uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
        si.∂u∂x, si.∂u∂y, si.∂v∂x, si.∂v∂y, ∂p∂x, ∂p∂y
    )
end

# Trace-specific spawn_ray functions for SurfaceInteraction
# Raycore has spawn_ray for its Interaction type, but we need these for our SurfaceInteraction
#
# Following pbrt-v4's SpawnRayTo: direction is (p1 - p0) unnormalized, so t=1 reaches p1.
# We use t_max = 1 - δ to stop just before p1, matching pbrt-v4's ShadowEpsilon behavior.
# This is critical for shadow rays where we only want to detect occluders BETWEEN p0 and p1.
@propagate_inbounds function spawn_ray(
        p0::Interaction, p1::Interaction, δ::Float32 = 1f-6,
    )::Ray
    direction = p1.p - p0.p
    origin = p0.p .+ δ .* direction
    # t_max = 1 - δ ensures we test for intersections between p0 and p1, not beyond
    return Ray(origin, direction, 1f0 - δ, p0.time)
end

@propagate_inbounds function spawn_ray(p0::SurfaceInteraction, p1::Interaction)::Ray
    spawn_ray(p0.core, p1)
end

@propagate_inbounds function spawn_ray(
        si::SurfaceInteraction, direction::Vec3f, δ::Float32 = 1f-6,
    )::Ray
    origin = si.core.p .+ δ .* direction
    return Ray(o=origin, d=direction, time=si.core.time)
end

# ============================================================================
# Shading coordinate system
# ============================================================================

sum_mul(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

"""
The shading coordinate system gives a frame for expressing directions
in spherical coordinates (θ, ϕ).
The angle θ is measured from the given direction to the z-axis
and ϕ is the angle formed with the x-axis after projection
of the direction onto xy-plane.

Since normal is `(0, 0, 1) → cos_θ = n · w = (0, 0, 1) ⋅ w = w.z`.
"""
@propagate_inbounds cos_θ(w::Vec3f) = w[3]
@propagate_inbounds sin_θ2(w::Vec3f) = max(0f0, 1f0 - cos_θ(w) * cos_θ(w))
@propagate_inbounds sin_θ(w::Vec3f) = √(sin_θ2(w))
@propagate_inbounds tan_θ(w::Vec3f) = sin_θ(w) / cos_θ(w)

@propagate_inbounds function cos_ϕ(w::Vec3f)
    sinθ = sin_θ(w)
    sinθ ≈ 0f0 ? 1f0 : clamp(w[1] / sinθ, -1f0, 1f0)
end
@propagate_inbounds function sin_ϕ(w::Vec3f)
    sinθ = sin_θ(w)
    sinθ ≈ 0f0 ? 1f0 : clamp(w[2] / sinθ, -1f0, 1f0)
end

function spherical_direction(sin_θ::Float32, cos_θ::Float32, ϕ::Float32)
    Vec3f(sin_θ * cos(ϕ), sin_θ * sin(ϕ), cos_θ)
end
function spherical_direction(
    sin_θ::Float32, cos_θ::Float32, ϕ::Float32,
    x::Vec3f, y::Vec3f, z::Vec3f,
)
    sin_θ * cos(ϕ) * x + sin_θ * sin(ϕ) * y + cos_θ * z
end

spherical_θ(v::Vec3f) = acos(clamp(v[3], -1f0, 1f0))
function spherical_ϕ(v::Vec3f)
    p = atan(v[2], v[1])
    p < 0 ? p + 2f0 * π : p
end

"""
Flip normal `n` so that it lies in the same hemisphere as `v`.
"""
@propagate_inbounds face_forward(n, v) = (n ⋅ v) < 0 ? -n : n

# ============================================================================
# Triangle intersection helpers
# ============================================================================

# Calculate partial derivatives for texture mapping
function partial_derivatives(vs::AbstractVector{Point3f}, uv::AbstractVector{Point2f})
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δp_13, δp_23 = Vec3f(vs[1] - vs[3]), Vec3f(vs[2] - vs[3])
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    if det ≈ 0f0
        v = normalize((vs[3] - vs[1]) × (vs[2] - vs[1]))
        ∂p∂u, ∂p∂v = coordinate_system(Vec3f(v))
        return ∂p∂u, ∂p∂v
    end
    inv_det = 1f0 / det
    ∂p∂u = Vec3f(δuv_23[2] * δp_13 - δuv_13[2] * δp_23) * inv_det
    ∂p∂v = Vec3f(-δuv_23[1] * δp_13 + δuv_13[1] * δp_23) * inv_det
    return ∂p∂u, ∂p∂v
end

# Convert Raycore.Triangle intersection result to Trace SurfaceInteraction
@propagate_inbounds function triangle_to_surface_interaction(triangle::Triangle, ray::AbstractRay, bary_coords::StaticVector{3,Float32})::SurfaceInteraction
    # Get triangle data
    verts = Raycore.vertices(triangle)
    tex_coords = Raycore.uvs(triangle)

    pos_deriv_u, pos_deriv_v = partial_derivatives(verts, tex_coords)

    # Interpolate hit point and texture coordinates using barycentric coordinates
    hit_point = sum_mul(bary_coords, verts)
    hit_uv = sum_mul(bary_coords, tex_coords)

    # Calculate surface normal from triangle edges
    edge1 = verts[2] - verts[1]
    edge2 = verts[3] - verts[1]
    normal = normalize(edge1 × edge2)

    # Create surface interaction data at hit point
    surf_interact = SurfaceInteraction(
        normal, hit_point, ray.time, -ray.d, hit_uv,
        pos_deriv_u, pos_deriv_v, Normal3f(0f0), Normal3f(0f0)
    )

    # Initialize shading geometry from triangle normals/tangents if available
    t_normals = Raycore.normals(triangle)
    t_tangents = Raycore.tangents(triangle)

    has_normals = !all(x -> all(isnan, x), t_normals)
    has_tangents = !all(x -> all(isnan, x), t_tangents)

    if !has_normals && !has_tangents
        return surf_interact
    end

    # Initialize shading normal
    shading_normal = surf_interact.core.n
    if has_normals
        shading_normal = normalize(sum_mul(bary_coords, t_normals))
    end

    # Calculate shading tangent
    shading_tangent = Vec3f(0)
    if has_tangents
        shading_tangent = normalize(sum_mul(bary_coords, t_tangents))
    else
        shading_tangent = normalize(pos_deriv_u)
    end

    # Calculate shading bitangent
    shading_bitangent = Vec3f(shading_normal × shading_tangent)

    if (shading_bitangent ⋅ shading_bitangent) > 0f0
        shading_bitangent = Vec3f(normalize(shading_bitangent))
        shading_tangent = Vec3f(shading_bitangent × shading_normal)
    else
        _, shading_tangent, shading_bitangent = coordinate_system(Vec3f(shading_normal))
    end

    return set_shading_geometry(
        surf_interact,
        shading_tangent,
        shading_bitangent,
        Normal3f(0f0),
        Normal3f(0f0),
        true
    )
end

# ============================================================================
# Accelerator intersection
# ============================================================================

# Intersect StaticTLAS - returns hit info, primitive, and SurfaceInteraction
# The primitive (triangle) contains material_type and material_idx for dispatch
# Note: Only StaticTLAS is used in kernels (adapt converts TLAS → StaticTLAS)
@propagate_inbounds function intersect!(accel::Raycore.StaticTLAS, ray::AbstractRay)
    hit_found, triangle, distance, bary_coords, instance_id = closest_hit(accel, ray)

    if !hit_found
        return false, triangle, SurfaceInteraction()
    end

    # Convert to SurfaceInteraction (in local/BLAS space)
    interaction = triangle_to_surface_interaction(triangle, ray, bary_coords)

    # Transform surface interaction to world space using instance transform
    # instance_id is 1-based array index into accel.instances (set during TLAS construction)
    # Use it directly instead of searching - this ensures we get the current transform
    # even after updates via update_transform!
    if instance_id >= 1 && instance_id <= length(accel.instances)
        inst = accel.instances[instance_id]
        transform = inst.transform
        inv_transform = inst.inv_transform

        # Transform hit point to world space
        local_p = interaction.core.p
        world_p = Point3f(transform * Vec4f(local_p..., 1f0))

        # Transform normal to world space: n_world = normalize(transpose(inv_transform) * n_local)
        # For a 4x4 matrix, we use the upper-left 3x3 for direction transforms
        local_n = Vec3f(interaction.core.n)
        # transpose(inv_transform) is equivalent to inverse-transpose of transform
        inv_t_3x3 = Mat3f(inv_transform[1,1], inv_transform[2,1], inv_transform[3,1],
                          inv_transform[1,2], inv_transform[2,2], inv_transform[3,2],
                          inv_transform[1,3], inv_transform[2,3], inv_transform[3,3])
        world_n = Normal3f(normalize(inv_t_3x3 * local_n))

        # Transform shading normal similarly
        local_sn = Vec3f(interaction.shading.n)
        world_sn = Normal3f(normalize(inv_t_3x3 * local_sn))

        # Transform tangent/bitangent (these transform like directions, using the forward transform)
        t_3x3 = Mat3f(transform[1,1], transform[2,1], transform[3,1],
                      transform[1,2], transform[2,2], transform[3,2],
                      transform[1,3], transform[2,3], transform[3,3])
        # ∂p∂u and ∂p∂v are on SurfaceInteraction directly, not on core
        world_dpdu = normalize(t_3x3 * interaction.∂p∂u)
        world_dpdv = normalize(t_3x3 * interaction.∂p∂v)
        # Shading tangent/bitangent
        world_st = normalize(t_3x3 * interaction.shading.∂p∂u)
        world_sb = normalize(t_3x3 * interaction.shading.∂p∂v)

        # Reconstruct SurfaceInteraction with world-space values
        # Interaction fields: p, time, wo, n
        core = Interaction(world_p, interaction.core.time, interaction.core.wo, world_n)
        # ShadingInteraction fields: n, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v
        shading = ShadingInteraction(world_sn, world_st, world_sb, interaction.shading.∂n∂u, interaction.shading.∂n∂v)
        # SurfaceInteraction constructor: core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂p∂x, ∂p∂y
        interaction = SurfaceInteraction(
            core, shading, interaction.uv,
            world_dpdu, world_dpdv, interaction.∂n∂u, interaction.∂n∂v,
            interaction.∂u∂x, interaction.∂u∂y, interaction.∂v∂x, interaction.∂v∂y,
            interaction.∂p∂x, interaction.∂p∂y
        )
    end

    return true, triangle, interaction
end

# Intersect BVH - returns hit info, primitive, and SurfaceInteraction
@propagate_inbounds function intersect!(accel::BVH, ray::AbstractRay)
    hit_found, triangle, distance, bary_coords = closest_hit(accel, ray)

    if !hit_found
        return false, triangle, SurfaceInteraction()
    end

    # Convert to SurfaceInteraction
    interaction = triangle_to_surface_interaction(triangle, ray, bary_coords)

    # Return primitive so caller can access triangle.metadata (SetKey)
    return true, triangle, interaction
end
