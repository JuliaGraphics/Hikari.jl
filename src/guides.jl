# Denoising guide extraction, shared by software and hardware primary queries.
# Guides describe the first opaque material, not emitted/illuminated radiance.
@inline function guide_rgb(value)
    c = rgb(value)
    return RGB{Float32}(clamp(c[1], 0.0f0, 1.0f0), clamp(c[2], 0.0f0, 1.0f0), clamp(c[3], 0.0f0, 1.0f0))
end
# White is the conventional fallback for specular/unknown BSDFs, not grey
# pretending to be diffuse reflectance.
@inline guide_material(mat::Material, materials, tfc, ::Val{D}) where {D} = RGB{Float32}(1, 1, 1)
@inline guide_material(mat::Diffuse, materials, tfc, ::Val{D}) where {D} =
    guide_rgb(eval_handle_spectrum(materials, mat.color, tfc))
@inline guide_material(mat::Union{Conductor, CoatedDiffuse, DiffuseTransmission, CoatedDiffuseTransmission},
                       materials, tfc, ::Val{D}) where {D} =
    guide_rgb(eval_handle_spectrum(materials, mat.reflectance, tfc))
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
        tfc = alpha_filter_context(prim, bary, ray.o, ray.d, t)
        # Deterministic centre-ray guides: skip null volume boundaries and
        # alpha-cutout surfaces below 50% coverage. No random guide flicker.
        alpha = surface_alpha_dispatch(materials, mi.material, tfc)
        if Raycore.is_valid(mi.material) && alpha >= 0.5f0
            geom = vp_compute_surface_geometry(prim, bary, ray.o, ray.d, t)
            color, ns = Raycore.with_index(material_guides, materials, mi.material,
                                           materials, tfc, Val(3), geom, prim, -ray.d)
            ns = dot(ns, ray.d) > 0.0f0 ? -ns : ns
            n2 = dot(ns, ns)
            ns = n2 > 0.0f0 ? ns / sqrt(n2) : Vec3f(0)
            # Film.depth also drives overlay/volume coverage. Preserve its
            # nearest-geometry meaning even when guide surfaces lie behind
            # transparent cutouts or null volume boundaries.
            return (color, ns, first_depth)
        end
        offset = t + max(1.0f-4, abs(t) * 1.0f-5)
        traveled += offset
        ray = Raycore.Ray(ray; o = Point3f(ray.o + offset * ray.d), t_min = 0.0f0)
    end
    return (RGB{Float32}(0, 0, 0), Vec3f(0), first_depth)
end

@inline function material_guides(mat::Material, materials, tfc, depth, geom, primitive, wo)
    dndu, dndv = vp_compute_normal_derivatives(primitive)
    ns, _ = perturb_shading_frame_impl(mat, materials, geom.ns, geom.dpdus, geom.dpdu, geom.dpdv,
                                       geom.n, () -> (dndu, dndv, tfc))
    return guide_material(mat, materials, tfc, depth), ns
end
@inline function material_guides(mat::ShaderMaterial{P, A, M}, materials, tfc, depth, geom, primitive, wo) where {P, A, M}
    sign = dot(cross(geom.ns, geom.dpdus), geom.dpdv) < 0f0 ? -1f0 : 1f0
    context = ShaderContext(tfc, wo, Vec3f(0), Vec3f(0), geom.dpdus, geom.n, sign,
                            Vec3f(0), Point2f(0), shader_geometry(mat.program, geom, primitive))
    albedo, normal = shader_guides(M, shader_programs(materials, mat.program)[mat.program.idx], materials, context, depth)
    return guide_rgb(RGBSpectrum(albedo[1], albedo[2], albedo[3])), normal == Vec3f(0) ? geom.ns : normal
end
@inline shader_guide_normal(w::Weighted) = w.sample_weight > 0f0 ? w.sample_weight * w.n : Vec3f(0)
@inline shader_guide_normal(w::PbrtTerm) = w.weight * first(pbrt_graph_frame(w.context))

const GUIDE_GRID = Int32(9)

"""
    pixel_guides(accel, materials, interfaces, camera, filter, filter_data, centre, miss_depth, grid) -> (albedo, normal, depth)

The guides of the pixel at raster position `centre`: albedo and normal over a `grid`²
stratified set of rays through `filter`'s samples, weighted as the image weights them, since
OIDN wants the guides filtered like the image, and the depth of the middle ray.
"""
@inline function pixel_guides(accel, materials, interfaces, camera, filter, filter_data, centre::Point2f,
                              miss_depth::Float32, grid::Int32)
    r = g = b = weights = 0.0f0
    normal = Vec3f(0)
    depth = miss_depth
    middle = (grid + Int32(1)) ÷ Int32(2)
    for j in Int32(1):grid, i in Int32(1):grid
        u = Point2f((i - 0.5f0) / grid, (j - 0.5f0) / grid)
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
