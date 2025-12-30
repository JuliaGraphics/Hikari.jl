import KernelAbstractions as KA

KA.@kernel some_kernel_f() = nothing

function to_gpu(ArrayType, m::Hikari.Texture)
    @assert !Hikari.no_texture(m)
    return Hikari.Texture(
        to_gpu(ArrayType, m.data),
        m.const_value,
        m.isconst,
    )
end

# GPU conversion for each material type

function to_gpu(ArrayType, m::Hikari.MatteMaterial)
    Kd = Hikari.no_texture(m.Kd) ? m.Kd : to_gpu(ArrayType, m.Kd)
    σ = Hikari.no_texture(m.σ) ? m.σ : to_gpu(ArrayType, m.σ)
    return Hikari.MatteMaterial(Kd, σ)
end

function to_gpu(ArrayType, m::Hikari.MirrorMaterial)
    Kr = Hikari.no_texture(m.Kr) ? m.Kr : to_gpu(ArrayType, m.Kr)
    return Hikari.MirrorMaterial(Kr)
end

function to_gpu(ArrayType, m::Hikari.GlassMaterial)
    Kr = Hikari.no_texture(m.Kr) ? m.Kr : to_gpu(ArrayType, m.Kr)
    Kt = Hikari.no_texture(m.Kt) ? m.Kt : to_gpu(ArrayType, m.Kt)
    u_roughness = Hikari.no_texture(m.u_roughness) ? m.u_roughness : to_gpu(ArrayType, m.u_roughness)
    v_roughness = Hikari.no_texture(m.v_roughness) ? m.v_roughness : to_gpu(ArrayType, m.v_roughness)
    index = Hikari.no_texture(m.index) ? m.index : to_gpu(ArrayType, m.index)
    return Hikari.GlassMaterial(Kr, Kt, u_roughness, v_roughness, index, m.remap_roughness)
end

function to_gpu(ArrayType, m::Hikari.PlasticMaterial)
    Kd = Hikari.no_texture(m.Kd) ? m.Kd : to_gpu(ArrayType, m.Kd)
    Ks = Hikari.no_texture(m.Ks) ? m.Ks : to_gpu(ArrayType, m.Ks)
    roughness = Hikari.no_texture(m.roughness) ? m.roughness : to_gpu(ArrayType, m.roughness)
    return Hikari.PlasticMaterial(Kd, Ks, roughness, m.remap_roughness)
end

function to_gpu(ArrayType, ms::Hikari.MaterialScene)
    accel = to_gpu(ArrayType, ms.accel)
    # Convert each material's textures to GPU, keep as tuple of vectors
    materials = map(ms.materials) do mats
        to_gpu(ArrayType, map(m -> to_gpu(ArrayType, m), mats))
    end
    return Hikari.MaterialScene(accel, materials)
end

# Helper to convert array to GPU device array (bitstype)
function array_to_device(ArrayType, arr::AbstractArray, preserve::Vector{Any})
    gpu_arr = ArrayType(arr)
    push!(preserve, gpu_arr)
    kernel = some_kernel_f(KA.get_backend(gpu_arr))
    return KA.argconvert(kernel, gpu_arr)
end

# GPU conversion for Distribution1D
function to_gpu(ArrayType, d::Hikari.Distribution1D, preserve::Vector{Any})
    func_gpu = array_to_device(ArrayType, d.func, preserve)
    cdf_gpu = array_to_device(ArrayType, d.cdf, preserve)
    return Hikari.Distribution1D(func_gpu, cdf_gpu, d.func_int)
end

# GPU conversion for Distribution2D
function to_gpu(ArrayType, d::Hikari.Distribution2D, preserve::Vector{Any})
    # Convert each conditional distribution to device arrays
    p_conditional_gpu = [to_gpu(ArrayType, p, preserve) for p in d.p_conditional_v]
    # The vector of Distribution1D structs also needs to be a device array
    p_conditional_vec = array_to_device(ArrayType, p_conditional_gpu, preserve)
    p_marginal_gpu = to_gpu(ArrayType, d.p_marginal, preserve)
    return Hikari.Distribution2D(p_conditional_vec, p_marginal_gpu)
end

# GPU conversion for EnvironmentMap
function to_gpu(ArrayType, env::Hikari.EnvironmentMap, preserve::Vector{Any})
    data_gpu = array_to_device(ArrayType, env.data, preserve)
    dist_gpu = to_gpu(ArrayType, env.distribution, preserve)
    return Hikari.EnvironmentMap(data_gpu, env.rotation, dist_gpu)
end

# GPU conversion for EnvironmentLight
function to_gpu(ArrayType, light::Hikari.EnvironmentLight, preserve::Vector{Any})
    env_map_gpu = to_gpu(ArrayType, light.env_map, preserve)
    return Hikari.EnvironmentLight(env_map_gpu, light.scale)
end

# GPU conversion for PointLight (already bitstype, no conversion needed)
to_gpu(::Type, light::Hikari.PointLight, ::Vector{Any}) = light

# GPU conversion for AmbientLight (already bitstype, no conversion needed)
to_gpu(::Type, light::Hikari.AmbientLight, ::Vector{Any}) = light

# Convert tuple of lights to GPU
to_gpu_lights(ArrayType, lights::Tuple, preserve::Vector{Any}) = map(l -> to_gpu(ArrayType, l, preserve), lights)

# Scene GPU conversion - returns (gpu_scene, preserve_array)
function to_gpu(ArrayType, scene::Hikari.Scene)
    aggregate = to_gpu(ArrayType, scene.aggregate)
    # Lights need special handling - store GPU arrays in preserve to keep them alive
    preserve = Any[]
    lights = to_gpu_lights(ArrayType, scene.lights, preserve)
    return Hikari.Scene(lights, aggregate, scene.bound, scene.world_center, scene.world_radius), preserve
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
    )
end
