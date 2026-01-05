import KernelAbstractions as KA

KA.@kernel some_kernel_f() = nothing

# Helper to check if a texture has actual data that needs GPU conversion
has_texture_data(t::Hikari.Texture) = isdefined(t, :data) && length(t.data) > 0

function to_gpu(ArrayType, m::Hikari.Texture{ElType, N, ArrType}) where {ElType, N, ArrType}
    if has_texture_data(m)
        # Convert actual texture data to GPU device array
        return Hikari.Texture(
            Raycore.to_gpu(ArrayType, m.data),
            m.const_value,
            m.isconst,
        )
    else
        # For constant textures, use SMatrix{0,0,T,0} as bitstype placeholder
        # No GPU memory needed - the const_value is used directly via dispatch
        return Hikari.Texture(SMatrix{0,0,ElType,0}(), m.const_value, true)
    end
end

# GPU conversion for each material type
# All textures are converted - to_gpu handles both data textures and constant textures

function to_gpu(ArrayType, m::Hikari.MatteMaterial)
    Kd = to_gpu(ArrayType, m.Kd)
    σ = to_gpu(ArrayType, m.σ)
    return Hikari.MatteMaterial(Kd, σ)
end

function to_gpu(ArrayType, m::Hikari.MirrorMaterial)
    Kr = to_gpu(ArrayType, m.Kr)
    return Hikari.MirrorMaterial(Kr)
end

function to_gpu(ArrayType, m::Hikari.GlassMaterial)
    Kr = to_gpu(ArrayType, m.Kr)
    Kt = to_gpu(ArrayType, m.Kt)
    u_roughness = to_gpu(ArrayType, m.u_roughness)
    v_roughness = to_gpu(ArrayType, m.v_roughness)
    index = to_gpu(ArrayType, m.index)
    return Hikari.GlassMaterial(Kr, Kt, u_roughness, v_roughness, index, m.remap_roughness)
end

function to_gpu(ArrayType, m::Hikari.PlasticMaterial)
    Kd = to_gpu(ArrayType, m.Kd)
    Ks = to_gpu(ArrayType, m.Ks)
    roughness = to_gpu(ArrayType, m.roughness)
    return Hikari.PlasticMaterial(Kd, Ks, roughness, m.remap_roughness)
end

function to_gpu(ArrayType, m::Hikari.MetalMaterial)
    eta = to_gpu(ArrayType, m.eta)
    k = to_gpu(ArrayType, m.k)
    roughness = to_gpu(ArrayType, m.roughness)
    reflectance = to_gpu(ArrayType, m.reflectance)
    return Hikari.MetalMaterial(eta, k, roughness, reflectance, m.remap_roughness)
end

function to_gpu(ArrayType, ms::Hikari.MaterialScene)
    accel = to_gpu(ArrayType, ms.accel)
    # Convert each material's textures to GPU, then convert the vector to device array
    # Must use Raycore.to_gpu for the vector conversion (returns device array via argconvert)
    materials = map(ms.materials) do mats
        gpu_mats = map(m -> to_gpu(ArrayType, m), mats)
        Raycore.to_gpu(ArrayType, gpu_mats)
    end
    return Hikari.MaterialScene(accel, materials)
end

# GPU conversion for Distribution1D - uses Raycore.to_gpu which handles preservation via global PRESERVE
function to_gpu(ArrayType, d::Hikari.Distribution1D)
    func_gpu = Raycore.to_gpu(ArrayType, d.func)
    cdf_gpu = Raycore.to_gpu(ArrayType, d.cdf)
    return Hikari.Distribution1D(func_gpu, cdf_gpu, d.func_int)
end

# GPU conversion for Distribution2D
function to_gpu(ArrayType, d::Hikari.Distribution2D)
    # Convert each conditional distribution to device arrays
    p_conditional_gpu = [to_gpu(ArrayType, p) for p in d.p_conditional_v]
    # The vector of Distribution1D structs also needs to be a device array
    p_conditional_vec = Raycore.to_gpu(ArrayType, p_conditional_gpu)
    p_marginal_gpu = to_gpu(ArrayType, d.p_marginal)
    return Hikari.Distribution2D(p_conditional_vec, p_marginal_gpu)
end

# GPU conversion for EnvironmentMap
function to_gpu(ArrayType, env::Hikari.EnvironmentMap)
    data_gpu = Raycore.to_gpu(ArrayType, env.data)
    dist_gpu = to_gpu(ArrayType, env.distribution)
    return Hikari.EnvironmentMap(data_gpu, env.rotation, dist_gpu)
end

# GPU conversion for EnvironmentLight
function to_gpu(ArrayType, light::Hikari.EnvironmentLight)
    env_map_gpu = to_gpu(ArrayType, light.env_map)
    return Hikari.EnvironmentLight(env_map_gpu, light.scale)
end

# GPU conversion for PointLight (already bitstype, no conversion needed)
to_gpu(::Type, light::Hikari.PointLight) = light

# GPU conversion for AmbientLight (already bitstype, no conversion needed)
to_gpu(::Type, light::Hikari.AmbientLight) = light

# Convert tuple of lights to GPU
to_gpu_lights(ArrayType, lights::Tuple) = map(l -> to_gpu(ArrayType, l), lights)

# Scene GPU conversion - uses Raycore.to_gpu which handles preservation via global PRESERVE
# Returns ImmutableScene for GPU (mutable structs can't be passed to GPU kernels)
function to_gpu(ArrayType, scene::Hikari.Scene)
    aggregate = to_gpu(ArrayType, scene.aggregate)
    lights = to_gpu_lights(ArrayType, scene.lights)
    return Hikari.ImmutableScene(lights, aggregate, scene.bound, scene.world_center, scene.world_radius)
end

@kernel function ka_trace_image!(img, camera, scene, sampler, max_depth)
    _idx = @index(Global)
    idx = _idx % Int32
    @_inbounds if checkbounds(Bool, img, idx)
        cols = size(img, 2) % Int32
        row = (idx - Int32(1)) ÷ cols + Int32(1)
        col = (idx - Int32(1)) % cols + Int32(1)
        pixel = Point2f((row, cols - col))
        l = trace_pixel(camera, scene, pixel, sampler, max_depth)
        img[idx] = RGB{Float32}(( l.c)...)
    end
    nothing
end

function launch_trace_image!(img, camera, scene, samples_per_pixel::Int32, max_depth::Int32, niter::Int32)
    backend = KA.get_backend(img)
    kernel! = ka_trace_image!(backend)
    sampler = UniformSampler(samples_per_pixel)
    kernel!(img, camera, scene, sampler, max_depth, niter, ndrange=size(img), workgroupsize=(16, 16))
    KA.synchronize(backend)
    return img
end


function Hikari.to_gpu(ArrayType, film::Film)
    return Film(
        film.resolution,
        film.crop_bounds,
        film.diagonal,
        KA.adapt(ArrayType, film.pixels),
        KA.adapt(ArrayType, film.tiles),
        film.tile_size,
        film.ntiles,
        film.filter_table,
        film.filter_table_width,
        film.filter_radius,
        film.scale,
        KA.adapt(ArrayType, film.framebuffer),
        KA.adapt(ArrayType, film.albedo),
        KA.adapt(ArrayType, film.normal),
        KA.adapt(ArrayType, film.depth),
        KA.adapt(ArrayType, film.postprocess),
    )
end
