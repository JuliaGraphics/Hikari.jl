module SubsurfaceSetupProbe
using Hikari, Mantle, KernelAbstractions, GeometryBasics, JSON3
using LinearAlgebra: normalize
using Raycore

@kernel function walk_probe!(out, ::Type{P}, programs, index, materials, table, dummy, accel, interfaces) where {P}
    j = @index(Global)
    n = Vec3f(0, -1, 0)
    p = Point3f(0, -1, 0)
    du, dv = Vec3f(1, 0, 0), Vec3f(0, 0, 1)
    lambda = Hikari.sample_wavelengths_visible(0.17f0)
    sc = Hikari.ShaderContext(Hikari.TextureFilterContext(Point2f(0), p, n), n,
        Vec3f(0), Vec3f(0), du, n, 1f0, Vec3f(0), Point2f(0))
    b = Hikari.surface_program_bsdf(P, Hikari.GatheredStack(dummy, Int32(1), Int32(1)),
        programs[index], sc, Hikari.LobeSetup(Hikari.lookup_tables(materials), false), table, lambda)
    ray = Raycore.Ray(o=Point3f(0,-4,0),d=-n,t_max=Inf32)
    rw = Hikari.VPRayWorkItem(ray, lambda, Int32(j))
    mi = interfaces[UInt32(1)]
    hw = Hikari.VPHitSurfaceWorkItem(rw, p, p + 1f-4 * n, n, du, dv, n, du, dv,
        Point2f(0), mi.material, mi, UInt32(1), Hikari.SVector{3,Float32}(1,0,0))
    work = Hikari.VPMaterialEvalWorkItem(hw, n, mi.material)
    exited, updated = Hikari.subsurface_scatter(b, work, accel, Int32(1))
    out[j] = exited.total
    f, pdf = Hikari.mixture(exited.lobes, updated.wo, updated.ns, updated.ns, lambda)
    out[256+j] = pdf
    out[512+j] = f[1]
    out[768+j] = updated.beta[1]
    rng = Hikari.pcg32_init(UInt64(j) << 32 | UInt64(1), UInt64(0x5353))
    u, _ = Hikari.pcg32_uniform_f32(rng)
    out[1024+j] = Float32(Hikari.subsurface_choice(b.lobes, u * b.total))
    out[1280+j] = b.total
    out[1536+j] = Hikari.subsurface_weight(b.lobes)
    out[1792+j] = Float32(exited.lobes.count)
end

@kernel function setup_probe!(out, ::Type{P}, programs, index, materials, table, dummy) where {P}
    n = normalize(Vec3f(dummy[1], dummy[2], dummy[3]))
    sc = Hikari.ShaderContext(Hikari.TextureFilterContext(Point2f(0), Point3f(0), n), n,
        Vec3f(0), Vec3f(0), Vec3f(1, 0, 0), n, 1f0, Vec3f(0), Point2f(0))
    lambda = Hikari.sample_wavelengths_visible(0.17f0)
    b = Hikari.surface_program_bsdf(P, Hikari.GatheredStack(dummy, Int32(1), Int32(1)),
        programs[index], sc, Hikari.LobeSetup(Hikari.lookup_tables(materials), false), table, lambda)
    out[1] = b.total
    out[2] = Hikari.subsurface_weight(b.lobes)
    out[3] = Float32(b.lobes.count)
    selected = Hikari.subsurface_choice(b.lobes, 0.5f0)
    out[4] = Float32(selected)
    params, weight = Hikari.picked_walk_params(b.lobes, selected, Val(1))
    out[5] = weight
    out[6] = params.radius[1]
    out[7] = params.albedo[1]
    out[8] = params.ior
    out[9] = params.n[1]
    out[10] = params.n[2]
    out[11] = params.n[3]
end

function run(vp, output)
    device = vp.state.memory.dev
    materials = vp.adapted.materials
    material_type = eltype(first(materials.data))
    P = Hikari.bsdf_kernel_nodes(Hikari.bsdf_nodes(material_type), Mantle.syncbackend(device))
    program_index = first(Array(vp.state.shader_inputs.program))
    programs = Hikari.closure_programs(materials)
    out = Mantle.devicearray(device, zeros(Float32, 11))
    dummy = Mantle.devicearray(device, Float32[0.2, 0.3, 1])
    setup_probe!(device, 1)(out, P, programs, program_index, Hikari.closure_set(materials), vp.state.rgb2spec_table, dummy; ndrange = 1)
    Mantle.waitidle(device)
    actual = Array(out)
    walks = Mantle.devicearray(device, zeros(Float32, 2048))
    walk_probe!(device, 32)(walks, P, programs, program_index, Hikari.closure_set(materials),
        vp.state.rgb2spec_table, dummy, vp.adapted.accel, vp.adapted.media_interfaces; ndrange = 256)
    Mantle.waitidle(device)
    walk_values = Array(walks)
    open(io -> JSON3.write(io, (; device = Mantle.devicename(device), actual, walk_values)), output, "w")
    println("SETUP_CONTROL ", actual); flush(stdout)
    println("WALK_CONTROL surviving ", count(>(0f0), walk_values[1:256]), "/256; pdf positive ",
        count(>(0f0), walk_values[257:512]), "; beta range ", extrema(walk_values[769:1024])); flush(stdout)
    nothing
end
end
