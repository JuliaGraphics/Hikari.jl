import KernelAbstractions as KA

KA.@kernel some_kernel_f() = nothing

function some_kernel(arr)
    backend = KA.get_backend(arr)
    return some_kernel_f(backend)
end

function to_gpu(ArrayType, m::AbstractArray; preserve=[])
    arr = ArrayType(m)
    push!(preserve, arr)
    kernel = some_kernel(arr)
    return KA.argconvert(kernel, arr)
end

function to_gpu(ArrayType, m::Hikari.Texture; preserve=[])
    @assert !Hikari.no_texture(m)
    return Hikari.Texture(
        to_gpu(ArrayType, m.data; preserve=preserve),
        m.const_value,
        m.isconst,
    )
end

# GPU conversion for each material type (with preserve for OpenCL)

function to_gpu(ArrayType, m::Hikari.MatteMaterial; preserve=[])
    Kd = Hikari.no_texture(m.Kd) ? m.Kd : to_gpu(ArrayType, m.Kd; preserve=preserve)
    σ = Hikari.no_texture(m.σ) ? m.σ : to_gpu(ArrayType, m.σ; preserve=preserve)
    return Hikari.MatteMaterial(Kd, σ)
end

function to_gpu(ArrayType, m::Hikari.MirrorMaterial; preserve=[])
    Kr = Hikari.no_texture(m.Kr) ? m.Kr : to_gpu(ArrayType, m.Kr; preserve=preserve)
    return Hikari.MirrorMaterial(Kr)
end

function to_gpu(ArrayType, m::Hikari.GlassMaterial; preserve=[])
    Kr = Hikari.no_texture(m.Kr) ? m.Kr : to_gpu(ArrayType, m.Kr; preserve=preserve)
    Kt = Hikari.no_texture(m.Kt) ? m.Kt : to_gpu(ArrayType, m.Kt; preserve=preserve)
    u_roughness = Hikari.no_texture(m.u_roughness) ? m.u_roughness : to_gpu(ArrayType, m.u_roughness; preserve=preserve)
    v_roughness = Hikari.no_texture(m.v_roughness) ? m.v_roughness : to_gpu(ArrayType, m.v_roughness; preserve=preserve)
    index = Hikari.no_texture(m.index) ? m.index : to_gpu(ArrayType, m.index; preserve=preserve)
    return Hikari.GlassMaterial(Kr, Kt, u_roughness, v_roughness, index, m.remap_roughness)
end

function to_gpu(ArrayType, m::Hikari.PlasticMaterial; preserve=[])
    Kd = Hikari.no_texture(m.Kd) ? m.Kd : to_gpu(ArrayType, m.Kd; preserve=preserve)
    Ks = Hikari.no_texture(m.Ks) ? m.Ks : to_gpu(ArrayType, m.Ks; preserve=preserve)
    roughness = Hikari.no_texture(m.roughness) ? m.roughness : to_gpu(ArrayType, m.roughness; preserve=preserve)
    return Hikari.PlasticMaterial(Kd, Ks, roughness, m.remap_roughness)
end

# Conversion constructor for e.g. GPU arrays
# TODO, create tree on GPU? Not sure if that will gain much though...
function to_gpu(ArrayType, bvh::Hikari.BVH; preserve=[])
    primitives = to_gpu(ArrayType, bvh.primitives; preserve=preserve)
    nodes = to_gpu(ArrayType, bvh.nodes; preserve=preserve)
    materials = to_gpu(ArrayType, to_gpu.((ArrayType,), bvh.materials; preserve=preserve); preserve=preserve)
    return Hikari.BVH(primitives, materials, bvh.max_node_primitives, nodes)
end

function to_gpu(ArrayType, scene::Hikari.Scene; preserve=[])
    aggregate = to_gpu(ArrayType, scene.aggregate; preserve=preserve)
    lights = to_gpu_lights(ArrayType, scene.lights; preserve=preserve)
    return Hikari.Scene(lights, aggregate, scene.bound, scene.world_center, scene.world_radius)
end

# GPU conversion for Distribution1D
function to_gpu(ArrayType, d::Hikari.Distribution1D; preserve=[])
    func_gpu = to_gpu(ArrayType, d.func; preserve=preserve)
    cdf_gpu = to_gpu(ArrayType, d.cdf; preserve=preserve)
    return Hikari.Distribution1D(func_gpu, cdf_gpu, d.func_int)
end

# GPU conversion for Distribution2D
function to_gpu(ArrayType, d::Hikari.Distribution2D; preserve=[])
    # Convert each conditional distribution
    p_conditional_gpu = [to_gpu(ArrayType, p; preserve=preserve) for p in d.p_conditional_v]
    p_conditional_vec = to_gpu(ArrayType, p_conditional_gpu; preserve=preserve)
    p_marginal_gpu = to_gpu(ArrayType, d.p_marginal; preserve=preserve)
    return Hikari.Distribution2D(p_conditional_vec, p_marginal_gpu)
end

# GPU conversion for EnvironmentMap
function to_gpu(ArrayType, env::Hikari.EnvironmentMap; preserve=[])
    data_gpu = to_gpu(ArrayType, env.data; preserve=preserve)
    dist_gpu = to_gpu(ArrayType, env.distribution; preserve=preserve)
    return Hikari.EnvironmentMap(data_gpu, env.rotation, dist_gpu)
end

# GPU conversion for EnvironmentLight
function to_gpu(ArrayType, light::Hikari.EnvironmentLight; preserve=[])
    env_map_gpu = to_gpu(ArrayType, light.env_map; preserve=preserve)
    return Hikari.EnvironmentLight(env_map_gpu, light.scale)
end

# GPU conversion for PointLight (already bitstype, no conversion needed)
function to_gpu(::Type, light::Hikari.PointLight; preserve=[])
    return light
end

# GPU conversion for SunSkyLight
function to_gpu(ArrayType, light::Hikari.SunSkyLight; preserve=[])
    dist_gpu = to_gpu(ArrayType, light.distribution; preserve=preserve)
    return Hikari.SunSkyLight(
        light.sun_direction,
        light.sun_intensity,
        light.sun_angular_radius,
        light.turbidity,
        light.ground_albedo,
        light.ground_enabled,
        light.perez_Y,
        light.perez_x,
        light.perez_y,
        light.zenith_Y,
        light.zenith_x,
        light.zenith_y,
        dist_gpu,
    )
end

# Convert tuple of lights to GPU
function to_gpu_lights(ArrayType, lights::Tuple; preserve=[])
    return map(l -> to_gpu(ArrayType, l; preserve=preserve), lights)
end
