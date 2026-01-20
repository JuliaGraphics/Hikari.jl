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

# ============================================================================
# Legacy to_gpu for materials (converts textures to GPU arrays)
# These are deprecated - use to_gpu_with_collector instead for the TextureRef pattern
# ============================================================================

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

function to_gpu(ArrayType, m::Hikari.MetalMaterial)
    eta = to_gpu(ArrayType, m.eta)
    k = to_gpu(ArrayType, m.k)
    roughness = to_gpu(ArrayType, m.roughness)
    reflectance = to_gpu(ArrayType, m.reflectance)
    return Hikari.MetalMaterial(eta, k, roughness, reflectance, m.remap_roughness)
end

# GPU conversion for MediumInterface - convert wrapped material to GPU
# MediumIndex values are just Int32, so they transfer directly
function to_gpu(ArrayType, m::Hikari.MediumInterface)
    gpu_material = to_gpu(ArrayType, m.material)
    return Hikari.MediumInterface(gpu_material, m.inside, m.outside)
end

# GPU conversion for MediumInterfaceIdx - convert wrapped material to GPU
# MediumIndex values are just Int32 bitstype, so they transfer directly
function to_gpu(ArrayType, m::Hikari.MediumInterfaceIdx)
    gpu_material = to_gpu(ArrayType, m.material)
    return Hikari.MediumInterfaceIdx(gpu_material, m.inside, m.outside)
end

# ============================================================================
# TextureRef-based GPU conversion (new pattern)
# Converts Texture -> TextureRef using a TextureCollector
# ============================================================================

"""
    to_gpu_ref(collector::TextureCollector, m::Material) -> Material with TextureRef

Convert a material to use TextureRef instead of Texture, registering non-constant
textures in the collector for later GPU transfer.
"""
function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.MatteMaterial)
    Kd = Hikari.texture_to_ref(m.Kd, collector)
    σ = Hikari.texture_to_ref(m.σ, collector)
    return Hikari.MatteMaterial(Kd, σ)
end

function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.MirrorMaterial)
    Kr = Hikari.texture_to_ref(m.Kr, collector)
    return Hikari.MirrorMaterial(Kr)
end

function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.GlassMaterial)
    Kr = Hikari.texture_to_ref(m.Kr, collector)
    Kt = Hikari.texture_to_ref(m.Kt, collector)
    u_roughness = Hikari.texture_to_ref(m.u_roughness, collector)
    v_roughness = Hikari.texture_to_ref(m.v_roughness, collector)
    index = Hikari.texture_to_ref(m.index, collector)
    return Hikari.GlassMaterial(Kr, Kt, u_roughness, v_roughness, index, m.remap_roughness)
end

function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.MetalMaterial)
    eta = Hikari.texture_to_ref(m.eta, collector)
    k = Hikari.texture_to_ref(m.k, collector)
    roughness = Hikari.texture_to_ref(m.roughness, collector)
    reflectance = Hikari.texture_to_ref(m.reflectance, collector)
    return Hikari.MetalMaterial(eta, k, roughness, reflectance, m.remap_roughness)
end

function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.EmissiveMaterial)
    Le = Hikari.texture_to_ref(m.Le, collector)
    return Hikari.EmissiveMaterial(Le, m.scale, m.two_sided)
end

function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.CoatedDiffuseMaterial)
    reflectance = Hikari.texture_to_ref(m.reflectance, collector)
    u_roughness = Hikari.texture_to_ref(m.u_roughness, collector)
    v_roughness = Hikari.texture_to_ref(m.v_roughness, collector)
    thickness = Hikari.texture_to_ref(m.thickness, collector)
    albedo = Hikari.texture_to_ref(m.albedo, collector)
    g = Hikari.texture_to_ref(m.g, collector)
    return Hikari.CoatedDiffuseMaterial(
        reflectance, u_roughness, v_roughness, thickness,
        m.eta, albedo, g, m.max_depth, m.n_samples, m.remap_roughness
    )
end

# ThinDielectricMaterial - no textures, just scalar eta
function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.ThinDielectricMaterial)
    return m  # No textures to convert
end

# DiffuseTransmissionMaterial - reflectance and transmittance textures
function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.DiffuseTransmissionMaterial)
    reflectance = Hikari.texture_to_ref(m.reflectance, collector)
    transmittance = Hikari.texture_to_ref(m.transmittance, collector)
    return Hikari.DiffuseTransmissionMaterial(reflectance, transmittance, m.scale)
end

# CoatedConductorMaterial - many texture fields
function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.CoatedConductorMaterial)
    interface_u_roughness = Hikari.texture_to_ref(m.interface_u_roughness, collector)
    interface_v_roughness = Hikari.texture_to_ref(m.interface_v_roughness, collector)
    conductor_eta = Hikari.texture_to_ref(m.conductor_eta, collector)
    conductor_k = Hikari.texture_to_ref(m.conductor_k, collector)
    reflectance = Hikari.texture_to_ref(m.reflectance, collector)
    conductor_u_roughness = Hikari.texture_to_ref(m.conductor_u_roughness, collector)
    conductor_v_roughness = Hikari.texture_to_ref(m.conductor_v_roughness, collector)
    thickness = Hikari.texture_to_ref(m.thickness, collector)
    albedo = Hikari.texture_to_ref(m.albedo, collector)
    g = Hikari.texture_to_ref(m.g, collector)
    return Hikari.CoatedConductorMaterial(
        interface_u_roughness, interface_v_roughness, m.interface_eta,
        conductor_eta, conductor_k, reflectance,
        conductor_u_roughness, conductor_v_roughness,
        thickness, albedo, g,
        m.max_depth, m.n_samples, m.remap_roughness, m.use_eta_k
    )
end

# MediumInterface - convert wrapped material
function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.MediumInterface)
    gpu_material = to_gpu_ref(collector, m.material)
    return Hikari.MediumInterface(gpu_material, m.inside, m.outside)
end

function to_gpu_ref(collector::Hikari.TextureCollector, m::Hikari.MediumInterfaceIdx)
    gpu_material = to_gpu_ref(collector, m.material)
    return Hikari.MediumInterfaceIdx(gpu_material, m.inside, m.outside)
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

function to_gpu(ArrayType, ms::Hikari.MaterialScene)
    accel = to_gpu(ArrayType, ms.accel)

    # Create texture collector to gather all textures from materials
    collector = Hikari.TextureCollector()

    # Convert materials to use TextureRef (stores indices into texture tuple)
    # All materials of the same type will have identical TextureRef types (just different indices)
    materials = map(ms.materials) do mats
        convert_materials_to_gpu_ref(ArrayType, collector, mats)
    end

    # Build texture tuple from collected textures and convert to GPU
    # cpu_textures is a tuple of vectors, where each vector contains raw data matrices
    # (NOT Texture structs). This avoids SPIR-V pointer-in-composite issues.
    #
    # Structure: (Vector{Matrix{RGBSpectrum}}, Vector{Matrix{Float32}}, ...)
    # Each vector becomes a device array of device matrices on GPU.
    cpu_textures = Hikari.build_texture_tuple(collector)
    textures = map(cpu_textures) do data_vec
        # Each element is a Vector of data matrices - convert to device array
        # First convert each matrix to GPU, then wrap in device array
        gpu_matrices = [to_gpu(ArrayType, mat) for mat in data_vec]
        # Convert the vector of GPU matrices to a device array
        Raycore.to_gpu(ArrayType, gpu_matrices)
    end

    # Convert media tuple to GPU (each medium may have arrays like density grids)
    media = map(m -> to_gpu(ArrayType, m), ms.media)

    return Hikari.MaterialScene(accel, materials, media, textures)
end

"""
Convert a vector of materials to GPU using TextureRef pattern.
All materials get converted to use TextureRef, registering textures in the collector.
This ensures all materials of the same base type have identical concrete types.
"""
function convert_materials_to_gpu_ref(ArrayType, collector::Hikari.TextureCollector, mats::Vector{M}) where M
    isempty(mats) && return Raycore.to_gpu(ArrayType, Vector{Any}())

    # Convert all materials to TextureRef form
    ref_mats = [to_gpu_ref(collector, m) for m in mats]

    # All materials should now have the same type (TextureRef fields are uniform)
    T1 = typeof(first(ref_mats))
    typed_mats = T1[m for m in ref_mats]

    return Raycore.to_gpu(ArrayType, typed_mats)
end

# Fallback for non-vector material containers
function convert_materials_to_gpu_ref(ArrayType, collector::Hikari.TextureCollector, mats)
    ref_mats = map(m -> to_gpu_ref(collector, m), mats)
    Raycore.to_gpu(ArrayType, collect(ref_mats))
end

"""
Convert a vector of materials to GPU, ensuring homogeneous element types.

The challenge is that materials with different texture configurations (some with actual
texture data, some constant) become different concrete types after GPU conversion.
GPU arrays require a single concrete element type.

Solution: Convert all materials, then create a new vector with the widest compatible type
by converting constant textures to use 1-element GPU arrays when needed for consistency.
"""
function convert_materials_to_gpu(ArrayType, mats::Vector{M}) where M
    isempty(mats) && return Raycore.to_gpu(ArrayType, M[])

    # Convert all materials first
    gpu_mats_raw = [to_gpu(ArrayType, m) for m in mats]

    # Check if all have the same type (common case - homogeneous materials)
    T1 = typeof(first(gpu_mats_raw))
    all_same = all(m -> typeof(m) === T1, gpu_mats_raw)

    if all_same
        # Fast path: all same type, just create typed vector
        gpu_mats = T1[m for m in gpu_mats_raw]
        return Raycore.to_gpu(ArrayType, gpu_mats)
    end

    # Slow path: heterogeneous types - need to unify
    # This happens when some materials have texture data and others are constant.
    # We need to promote constant textures to use minimal GPU arrays for type consistency.
    gpu_mats_unified = unify_material_types(ArrayType, gpu_mats_raw)
    return Raycore.to_gpu(ArrayType, gpu_mats_unified)
end

# Fallback for non-vector material containers
function convert_materials_to_gpu(ArrayType, mats)
    gpu_mats = map(m -> to_gpu(ArrayType, m), mats)
    Raycore.to_gpu(ArrayType, collect(gpu_mats))
end

"""
Unify heterogeneous material types by promoting constant textures to minimal GPU arrays.
"""
function unify_material_types(ArrayType, gpu_mats::Vector)
    # For MatteMaterial: promote SMatrix{0,0} textures to 1-element GPU arrays
    # This ensures all materials have the same concrete type
    if !isempty(gpu_mats) && first(gpu_mats) isa Hikari.MatteMaterial
        return unify_matte_materials(ArrayType, gpu_mats)
    end
    # For other material types, try a similar approach
    # For now, fall back to Any[] which won't work on GPU but will error clearly
    @warn "Heterogeneous material types detected - GPU conversion may fail" types=unique(typeof.(gpu_mats))
    return gpu_mats
end

function unify_matte_materials(ArrayType, gpu_mats::Vector)
    # Find the "widest" texture types (prefer GPU arrays over SMatrix{0,0})
    # We'll promote all constant textures to use minimal 1-element GPU arrays

    # Determine target types from materials that have actual texture data
    KdType = nothing
    σType = nothing

    for m in gpu_mats
        Kd_data_type = typeof(m.Kd.data)
        σ_data_type = typeof(m.σ.data)

        # Prefer non-SMatrix types (actual GPU arrays)
        if !(Kd_data_type <: StaticArraysCore.SMatrix) && KdType === nothing
            KdType = Kd_data_type
        end
        if !(σ_data_type <: StaticArraysCore.SMatrix) && σType === nothing
            σType = σ_data_type
        end
    end

    # If no GPU array types found, all are constant - just use the first type
    if KdType === nothing && σType === nothing
        T = typeof(first(gpu_mats))
        return T[m for m in gpu_mats]
    end

    # Create unified materials with consistent texture types
    unified = map(gpu_mats) do m
        Kd = promote_texture_type(ArrayType, m.Kd, KdType)
        σ = promote_texture_type(ArrayType, m.σ, σType)
        Hikari.MatteMaterial(Kd, σ)
    end

    T = typeof(first(unified))
    return T[m for m in unified]
end

"""
Promote a texture to use a specific array type if it's currently a constant (SMatrix{0,0}).
"""
function promote_texture_type(ArrayType, tex::Hikari.Texture{ElType}, TargetArrType) where ElType
    if TargetArrType === nothing
        # No target type - keep as is
        return tex
    end

    if typeof(tex.data) <: StaticArraysCore.SMatrix
        # Constant texture - promote to 1-element GPU array for type consistency
        # The texture is still marked as const, so lookup will use const_value
        dummy_arr = reshape([tex.const_value], 1, 1)
        gpu_arr = Raycore.to_gpu(ArrayType, dummy_arr)
        return Hikari.Texture(gpu_arr, tex.const_value, true)
    else
        # Already has GPU array data
        return tex
    end
end

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
function to_gpu(ArrayType, scene::Hikari.Scene)
    aggregate = to_gpu(ArrayType, scene.aggregate)
    lights = to_gpu_lights(ArrayType, scene.lights)
    return Hikari.ImmutableScene(lights, aggregate, scene.bound, scene.world_center, scene.world_radius)
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
