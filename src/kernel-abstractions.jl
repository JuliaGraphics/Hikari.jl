import KernelAbstractions as KA

function to_gpu(ArrayType, m::Hikari.Texture)
    @assert !Hikari.no_texture(m)
    return Hikari.Texture(
        to_gpu(ArrayType, m.data),
        m.const_value,
        m.isconst,
    )
end

function to_gpu(ArrayType, m::Hikari.UberMaterial)
    if !Hikari.no_texture(m.Kd)
        Kd = to_gpu(ArrayType, m.Kd)
        no_tex_s = typeof(Kd)()
        Kr = Hikari.no_texture(m.Kr) ? no_tex_s : to_gpu(ArrayType, m.Kr)
    else
        Kr = to_gpu(ArrayType, m.Kr)
        no_tex_s = typeof(Kr)()
        Kd = Hikari.no_texture(m.Kd) ? no_tex_s : to_gpu(ArrayType, m.Kd)
    end
    f_tex = to_gpu(ArrayType, Hikari.Texture(zeros(Float32, 1, 1)))
    no_tex_f = typeof(f_tex)()
    return Hikari.UberMaterial(
        Kd,
        Hikari.no_texture(m.Ks) ? no_tex_s : to_gpu(ArrayType, m.Ks),
        Hikari.no_texture(m.Kr) ? no_tex_s : to_gpu(ArrayType, m.Kr),
        Hikari.no_texture(m.Kt) ? no_tex_s : to_gpu(ArrayType, m.Kt), Hikari.no_texture(m.σ) ? no_tex_f : to_gpu(ArrayType, m.σ),
        Hikari.no_texture(m.roughness) ? no_tex_f : to_gpu(ArrayType, m.roughness),
        Hikari.no_texture(m.u_roughness) ? no_tex_f : to_gpu(ArrayType, m.u_roughness),
        Hikari.no_texture(m.v_roughness) ? no_tex_f : to_gpu(ArrayType, m.v_roughness),
        Hikari.no_texture(m.index) ? no_tex_f : to_gpu(ArrayType, m.index),
        m.remap_roughness,
        m.type,
    )
end

function to_gpu(ArrayType, ms::Hikari.MaterialScene)
    bvh = to_gpu(ArrayType, ms.bvh)
    # Convert each material's textures to GPU, keep as tuple of vectors
    materials = map(ms.materials) do mats
        to_gpu(ArrayType, map(m -> to_gpu(ArrayType, m), mats))
    end
    return Hikari.MaterialScene(bvh, materials)
end

function to_gpu(ArrayType, scene::Hikari.Scene)
    aggregate = to_gpu(ArrayType, scene.aggregate)
    return Hikari.Scene(scene.lights, aggregate, scene.bound, scene.world_center, scene.world_radius)
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
