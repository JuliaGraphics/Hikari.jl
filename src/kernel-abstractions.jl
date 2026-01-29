import KernelAbstractions as KA
import Adapt

KA.@kernel some_kernel_f() = nothing

# ============================================================================
# Adapt.adapt_structure methods for GPU conversion
# ============================================================================

# Texture - adapt data array
function Adapt.adapt_structure(to, m::Hikari.Texture)
    Hikari.Texture(Adapt.adapt(to, m.data))
end

# CloudVolume - adapt density array
function Adapt.adapt_structure(to, m::Hikari.CloudVolume)
    Hikari.CloudVolume(
        Adapt.adapt(to, m.density),
        m.origin,
        m.extent,
        m.extinction_scale,
        m.asymmetry_g,
        m.single_scatter_albedo
    )
end

# HomogeneousMedium - already bitstype, no adaptation needed
Adapt.adapt_structure(to, m::Hikari.HomogeneousMedium) = m

# MajorantGrid - adapt voxels array
function Adapt.adapt_structure(to, m::Hikari.MajorantGrid)
    Hikari.MajorantGrid(Adapt.adapt(to, m.voxels), m.res)
end

# GridMedium - adapt density and majorant_grid
function Adapt.adapt_structure(to, m::Hikari.GridMedium)
    Hikari.GridMedium(
        m.bounds,
        m.render_to_medium,
        m.medium_to_render,
        m.σ_a,
        m.σ_s,
        Adapt.adapt(to, m.density),
        m.density_res,
        m.g,
        Adapt.adapt(to, m.majorant_grid),
        m.max_density
    )
end

# RGBGridMedium - adapt optional grids and majorant_grid
function Adapt.adapt_structure(to, m::Hikari.RGBGridMedium)
    Hikari.RGBGridMedium(
        m.bounds,
        m.render_to_medium,
        m.medium_to_render,
        isnothing(m.σ_a_grid) ? nothing : Adapt.adapt(to, m.σ_a_grid),
        isnothing(m.σ_s_grid) ? nothing : Adapt.adapt(to, m.σ_s_grid),
        m.sigma_scale,
        isnothing(m.Le_grid) ? nothing : Adapt.adapt(to, m.Le_grid),
        m.Le_scale,
        m.grid_res,
        m.g,
        Adapt.adapt(to, m.majorant_grid)
    )
end

# NanoVDBMedium - adapt buffer and majorant_grid
function Adapt.adapt_structure(to, m::Hikari.NanoVDBMedium)
    Hikari.NanoVDBMedium(
        Adapt.adapt(to, m.buffer),
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
        Adapt.adapt(to, m.majorant_grid),
        m.max_density
    )
end

# Distribution1D - adapt func and cdf arrays
function Adapt.adapt_structure(to, d::Hikari.Distribution1D)
    Hikari.Distribution1D(
        Adapt.adapt(to, d.func),
        Adapt.adapt(to, d.cdf),
        d.func_int
    )
end

# Distribution2D - adapt all arrays
function Adapt.adapt_structure(to, d::Hikari.Distribution2D)
    Hikari.Distribution2D(
        Adapt.adapt(to, d.conditional_func),
        Adapt.adapt(to, d.conditional_cdf),
        Adapt.adapt(to, d.conditional_func_int),
        Adapt.adapt(to, d.marginal_func),
        Adapt.adapt(to, d.marginal_cdf),
        d.marginal_func_int,
        d.nu,
        d.nv
    )
end

# EnvironmentMap - adapt data and distribution
function Adapt.adapt_structure(to, env::Hikari.EnvironmentMap)
    Hikari.EnvironmentMap(
        Adapt.adapt(to, env.data),
        env.rotation,
        Adapt.adapt(to, env.distribution)
    )
end

# EnvironmentLight - adapt env_map
function Adapt.adapt_structure(to, light::Hikari.EnvironmentLight)
    Hikari.EnvironmentLight(
        Adapt.adapt(to, light.env_map),
        light.scale
    )
end

# Bitstype lights - no adaptation needed
Adapt.adapt_structure(to, light::Hikari.PointLight) = light
Adapt.adapt_structure(to, light::Hikari.AmbientLight) = light
Adapt.adapt_structure(to, light::Hikari.DirectionalLight) = light
Adapt.adapt_structure(to, light::Hikari.SunLight) = light

# SunSkyLight - adapt distribution
function Adapt.adapt_structure(to, light::Hikari.SunSkyLight)
    Hikari.SunSkyLight(
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
        Adapt.adapt(to, light.distribution),
    )
end


# Film - adapt pixel/tile arrays and framebuffers
function Adapt.adapt_structure(to, film::Film)
    Film(
        film.resolution,
        film.crop_bounds,
        film.diagonal,
        Adapt.adapt(to, film.pixels),
        Adapt.adapt(to, film.tiles),
        film.tile_size,
        film.ntiles,
        film.filter_table,
        film.filter_table_width,
        film.filter_radius,
        film.scale,
        Adapt.adapt(to, film.framebuffer),
        Adapt.adapt(to, film.albedo),
        Adapt.adapt(to, film.normal),
        Adapt.adapt(to, film.depth),
        Adapt.adapt(to, film.postprocess),
        film.iteration_index,  # RefValue is shared across CPU/GPU
    )
end

# ============================================================================
# Kernels and launch functions
# ============================================================================

@kernel inbounds=true function ka_trace_image!(img, camera, scene, sampler, max_depth)
    _idx = @index(Global)
    idx = _idx % Int32
    if checkbounds(Bool, img, idx)
        cols = size(img, 2) % Int32
        row = (idx - Int32(1)) ÷ cols + Int32(1)
        col = (idx - Int32(1)) % cols + Int32(1)
        pixel = Point2f((row, cols - col))
        l = trace_pixel(camera, scene, pixel, sampler, max_depth)
        img[idx] = RGB{Float32}((l.c)...)
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
