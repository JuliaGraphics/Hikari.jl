# Denoising guide extraction, shared by software and hardware primary queries.
# Guides describe the first opaque material, not emitted/illuminated radiance.
@inline function guide_rgb(value)
    c = rgb(value)
    return RGB{Float32}(clamp(c[1], 0.0f0, 1.0f0), clamp(c[2], 0.0f0, 1.0f0), clamp(c[3], 0.0f0, 1.0f0))
end
@inline function guide_material(mat::Material, materials, tfc, ::Val{D}) where {D}
    # White is the conventional fallback for specular/unknown BSDFs, not grey
    # pretending to be diffuse reflectance.
    if hasfield(typeof(mat), :Kd)
        return guide_rgb(eval_handle_spectrum(materials, mat.Kd, tfc))
    elseif hasfield(typeof(mat), :reflectance)
        return guide_rgb(eval_handle_spectrum(materials, mat.reflectance, tfc))
    end
    return RGB{Float32}(1, 1, 1)
end
@inline function guide_material(mat::MixMaterial, materials, tfc, ::Val{D}) where {D}
    D == 0 && return RGB{Float32}(1, 1, 1)
    amount = clamp(eval_handle(materials, mat.amount, tfc), 0.0f0, 1.0f0)
    a = guide_dispatch(materials, mat.material1_idx, tfc, Val(D - 1))
    b = guide_dispatch(materials, mat.material2_idx, tfc, Val(D - 1))
    return RGB{Float32}(a.r * (1 - amount) + b.r * amount, a.g * (1 - amount) + b.g * amount, a.b * (1 - amount) + b.b * amount)
end
@inline function guide_dispatch(materials, key, tfc, depth)
    Raycore.is_valid(key) || return RGB{Float32}(0, 0, 0)
    return Raycore.with_index(guide_material, materials, key, materials, tfc, depth)
end

@inline function guide_hit(accel, materials, interfaces, ray, miss_depth)
    traveled = 0.0f0
    first_depth = miss_depth
    for step in 1:32
        hit, prim, t, bary, instance = Raycore.closest_hit(accel, ray)
        hit || return (RGB{Float32}(0, 0, 0), Vec3f(0), first_depth)
        step == 1 && (first_depth = t)
        mi = interfaces[resolve_mi_idx(instance, prim)]
        uv = vp_compute_uv_barycentric(prim, bary)
        tfc = TextureFilterContext(
            uv, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
            prim.metadata.primitive_index, SVector{3, Float32}(bary)
        )
        # Deterministic centre-ray guides: skip null volume boundaries and
        # alpha-cutout surfaces below 50% coverage. No random guide flicker.
        alpha = get_surface_alpha_dispatch(materials, mi.material, tfc)
        if Raycore.is_valid(mi.material) && alpha >= 0.5f0
            geom = vp_compute_surface_geometry(prim, bary, ray.o, ray.d, t)
            dndu, dndv = vp_compute_normal_derivatives(prim)
            ns, _ = get_perturbed_shading_frame(
                materials, mi.material, geom.ns, geom.dpdus,
                geom.dpdu, geom.dpdv, dndu, dndv, geom.n, tfc
            )
            ns = dot(ns, ray.d) > 0.0f0 ? -ns : ns
            n2 = dot(ns, ns)
            ns = n2 > 0.0f0 ? ns / sqrt(n2) : Vec3f(0)
            # Film.depth also drives overlay/volume coverage. Preserve its
            # nearest-geometry meaning even when guide surfaces lie behind
            # transparent cutouts or null volume boundaries.
            return (guide_dispatch(materials, mi.material, tfc, Val(3)), ns, first_depth)
        end
        offset = t + max(1.0f-4, abs(t) * 1.0f-5)
        traveled += offset
        ray = Raycore.Ray(ray; o = Point3f(ray.o + offset * ray.d), t_min = 0.0f0)
    end
    return (RGB{Float32}(0, 0, 0), Vec3f(0), first_depth)
end

const GUIDE_GRID = Int32(9)

"""
    pixel_guides(accel, materials, interfaces, camera, filter, filter_data, centre, miss_depth) -> (albedo, normal, depth)

The guides of the pixel at raster position `centre`: albedo and normal over a `GUIDE_GRID`²
stratified set of rays through `filter`'s samples, weighted as the image weights them, since
OIDN wants the guides filtered like the image, and the depth of the middle ray.
"""
@inline function pixel_guides(accel, materials, interfaces, camera, filter, filter_data, centre::Point2f, miss_depth::Float32)
    r = g = b = weights = 0.0f0
    normal = Vec3f(0)
    depth = miss_depth
    middle = (GUIDE_GRID + Int32(1)) ÷ Int32(2)
    for j in Int32(1):GUIDE_GRID, i in Int32(1):GUIDE_GRID
        u = Point2f((i - 0.5f0) / GUIDE_GRID, (j - 0.5f0) / GUIDE_GRID)
        fs = filter_sample(filter, filter_data, u)
        ray, ω = generate_ray(camera, CameraSample(centre + fs.p, Point2f(0.5f0, 0.5f0), 0.0f0))
        ω > 0.0f0 || continue
        a, n, d = guide_hit(accel, materials, interfaces, ray, miss_depth)
        w = fs.weight
        r += w * a.r; g += w * a.g; b += w * a.b
        normal += w * n
        weights += w
        i == j == middle && (depth = d)
    end
    k = weights > 0.0f0 ? 1.0f0 / weights : 0.0f0
    return (RGB{Float32}(r * k, g * k, b * k), normal * k, depth)
end
