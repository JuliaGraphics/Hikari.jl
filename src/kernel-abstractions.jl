import KernelAbstractions as KA

KA.@kernel some_kernel_f() = nothing

# GPU conversion for Texture - just convert the data array
function to_gpu(ArrayType, m::Hikari.Texture)
    return Hikari.Texture(Raycore.to_gpu(ArrayType, m.data))
end

# GPU conversion for CloudVolume - convert density array to GPU
function to_gpu(ArrayType, m::Hikari.CloudVolume)
    density_gpu = Raycore.to_gpu(ArrayType, m.density)
    return Hikari.CloudVolume(
        density_gpu,
        m.origin,
        m.extent,
        m.extinction_scale,
        m.asymmetry_g,
        m.single_scatter_albedo
    )
end

# GPU conversion for HomogeneousMedium - all fields are bitstype, no conversion needed
to_gpu(::Type, m::Hikari.HomogeneousMedium) = m

# GPU conversion for GridMedium - convert density array and majorant grid to GPU
function to_gpu(ArrayType, m::Hikari.GridMedium)
    density_gpu = Raycore.to_gpu(ArrayType, m.density)
    majorant_voxels_gpu = Raycore.to_gpu(ArrayType, m.majorant_grid.voxels)
    majorant_grid_gpu = Hikari.MajorantGrid(majorant_voxels_gpu, m.majorant_grid.res)
    return Hikari.GridMedium(
        m.bounds,
        m.render_to_medium,
        m.medium_to_render,
        m.σ_a,
        m.σ_s,
        density_gpu,
        m.density_res,
        m.g,
        majorant_grid_gpu,
        m.max_density
    )
end

# GPU conversion for RGBGridMedium - convert RGB grids and majorant grid to GPU
function to_gpu(ArrayType, m::Hikari.RGBGridMedium)
    # Convert optional grids to GPU (nothing stays nothing)
    σ_a_grid_gpu = isnothing(m.σ_a_grid) ? nothing : Raycore.to_gpu(ArrayType, m.σ_a_grid)
    σ_s_grid_gpu = isnothing(m.σ_s_grid) ? nothing : Raycore.to_gpu(ArrayType, m.σ_s_grid)
    Le_grid_gpu = isnothing(m.Le_grid) ? nothing : Raycore.to_gpu(ArrayType, m.Le_grid)
    majorant_voxels_gpu = Raycore.to_gpu(ArrayType, m.majorant_grid.voxels)
    majorant_grid_gpu = Hikari.MajorantGrid(majorant_voxels_gpu, m.majorant_grid.res)
    return Hikari.RGBGridMedium(
        m.bounds,
        m.render_to_medium,
        m.medium_to_render,
        σ_a_grid_gpu,
        σ_s_grid_gpu,
        m.sigma_scale,
        Le_grid_gpu,
        m.Le_scale,
        m.grid_res,
        m.g,
        majorant_grid_gpu
    )
end

# GPU conversion for NanoVDBMedium - convert buffer and majorant grid to GPU
function to_gpu(ArrayType, m::Hikari.NanoVDBMedium)
    buffer_gpu = Raycore.to_gpu(ArrayType, m.buffer)
    majorant_voxels_gpu = Raycore.to_gpu(ArrayType, m.majorant_grid.voxels)
    majorant_grid_gpu = Hikari.MajorantGrid(majorant_voxels_gpu, m.majorant_grid.res)
    return Hikari.NanoVDBMedium(
        buffer_gpu,
        m.root_offset,
        m.upper_offset,
        m.lower_offset,
        m.leaf_offset,
        m.leaf_count,
        m.lower_count,
        m.upper_count,
        m.root_table_size,
        m.inv_mat,
        m.vec,
        m.bounds,
        m.index_bbox_min,
        m.index_bbox_max,
        m.σ_a,
        m.σ_s,
        m.g,
        majorant_grid_gpu,
        m.max_density
    )
end

# NOTE: MaterialScene GPU conversion will use MultiTypeVec pattern
# The old TextureCollector-based conversion has been removed

# GPU conversion for Distribution1D - uses Raycore.to_gpu which handles preservation via global PRESERVE
function to_gpu(ArrayType, d::Hikari.Distribution1D)
    func_gpu = Raycore.to_gpu(ArrayType, d.func)
    cdf_gpu = Raycore.to_gpu(ArrayType, d.cdf)
    return Hikari.Distribution1D(func_gpu, cdf_gpu, d.func_int)
end

# GPU conversion for Distribution2D -> FlatDistribution2D
# IMPORTANT: We convert to FlatDistribution2D to avoid nested device arrays
# which cause SPIR-V validation errors when pointers are extracted from
# structs loaded from device arrays and used in loops.
function to_gpu(ArrayType, d::Hikari.Distribution2D)
    # First flatten the distribution on CPU
    flat = Hikari.FlatDistribution2D(d)
    # Then convert to GPU
    return to_gpu(ArrayType, flat)
end

# GPU conversion for FlatDistribution2D
function to_gpu(ArrayType, d::Hikari.FlatDistribution2D)
    return Hikari.FlatDistribution2D(
        Raycore.to_gpu(ArrayType, d.conditional_func),
        Raycore.to_gpu(ArrayType, d.conditional_cdf),
        Raycore.to_gpu(ArrayType, d.conditional_func_int),
        Raycore.to_gpu(ArrayType, d.marginal_func),
        Raycore.to_gpu(ArrayType, d.marginal_cdf),
        d.marginal_func_int,
        d.nu,
        d.nv
    )
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

# GPU conversion for DirectionalLight (already bitstype, no conversion needed)
to_gpu(::Type, light::Hikari.DirectionalLight) = light

# GPU conversion for SunLight (already bitstype, no conversion needed)
to_gpu(::Type, light::Hikari.SunLight) = light

# GPU conversion for SunSkyLight - needs to convert the Distribution2D for importance sampling
function to_gpu(ArrayType, light::Hikari.SunSkyLight)
    dist_gpu = to_gpu(ArrayType, light.distribution)
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
to_gpu_lights(ArrayType, lights::Tuple) = map(l -> to_gpu(ArrayType, l), lights)

# Scene GPU conversion - uses Raycore.to_gpu which handles preservation via global PRESERVE
# Returns ImmutableScene for GPU (mutable structs can't be passed to GPU kernels)
# Materials and media MultiTypeVec are adapted via Adapt.jl (returns StaticMultiTypeVec)
function to_gpu(ArrayType, scene::Hikari.Scene)
    accel = to_gpu(ArrayType, scene.accel)
    lights = to_gpu_lights(ArrayType, scene.lights)
    # MultiTypeVec adapts to StaticMultiTypeVec via Adapt.jl
    materials = KA.adapt(ArrayType, scene.materials)
    media = KA.adapt(ArrayType, scene.media)
    return Hikari.ImmutableScene(lights, accel, materials, media, scene.bound, scene.world_center, scene.world_radius)
end

@kernel inbounds=true function ka_trace_image!(img, camera, scene, sampler, max_depth)
    _idx = @index(Global)
    idx = _idx % Int32
     if checkbounds(Bool, img, idx)
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
        film.iteration_index,  # RefValue is shared across CPU/GPU
    )
end
