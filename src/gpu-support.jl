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
    return Hikari.Scene(scene.lights, aggregate, scene.bound, scene.world_center, scene.world_radius)
end
