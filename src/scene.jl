# Scene type and operations

# Abstract scene type for dispatch - allows both mutable and immutable implementations
abstract type AbstractScene end

# Scene stores lights, accelerator, materials, and media directly.
# Materials and media use MultiTypeSet (or StaticMultiTypeSet for GPU/kernels).
# Textures are stored within the materials MultiTypeSet (materials have TextureRef fields).
# LightVec can be Tuple (for type-stable iteration) or AbstractVector.
struct Scene{Accel,LightVec,MatVec<:AbstractVector,MedVec<:AbstractVector, MI<:AbstractVector, Bounds} <: AbstractScene
    lights::LightVec
    accel::Accel  # BVH, TLAS, or any accelerator supporting closest_hit/any_hit
    materials::MatVec  # MultiTypeSet
    media::MedVec  # MultiTypeSet
    media_interfaces::MI  # MultiTypeSet
    bounds::Bounds  # RefValue{Tuple} - adapted to device RefValue for GPU
end

# Accessors for scene bounds (dereference the RefValue)
@inline world_bound(scene::Scene) = scene.bounds[][1]
@inline world_sphere(scene::Scene) = scene.bounds[][2]
@inline world_center(scene::Scene) = scene.bounds[][2].center
@inline world_radius(scene::Scene) = scene.bounds[][2].r

"""
    Scene(; backend=KA.CPU())

Create an empty mutable Scene for incremental construction.

# Example
```julia
scene = Scene()
push!(scene, mesh, material)           # Add geometry
push!(scene, PointLight(...))   # Add lights
push!(scene, AmbientLight(...))
sync!(scene)  # Build acceleration structure
```
"""
function Scene(; backend=KA.CPU())
    tlas = TLAS(backend)
    lights = MultiTypeSet(backend)
    materials = MultiTypeSet(backend)
    media = MultiTypeSet(backend)
    media_interfaces = KA.allocate(backend, MediumInterfaceIdx, 0)
    Scene(lights, tlas, materials, media, media_interfaces, Ref((Bounds3(), Sphere(Point3f(0), 0f0))))
end

function Scene(mesh_material_pairs::Vector{<:Tuple}; lights = (), backend = KA.CPU())
    scene = Scene(; backend=backend)
    for light in lights
        push!(scene, light)
    end
    foreach(mesh_material_pairs) do args
        return push!(scene, args...)
    end
    # Build the BVH structure
    sync!(scene)
    return scene
end

# ============================================================================
# Scene push! methods
# ============================================================================

function Base.push!(scene::Scene, light::Light)
    push!(scene.lights, light)
end

function Base.push!(scene::Scene, material::Material)
    interface = MediumInterface(material)
    return push!(scene, interface)
end

function Base.push!(scene::Scene, medium::Medium)
    push!(scene.media, medium)
end

Base.push!(scene::Scene, medium::Nothing) = SetKey()

function Base.push!(scene::Scene, medium::MediumInterface)
    mat_idx = push!(scene.materials, medium.material)
    inside_idx = push!(scene, medium.inside)
    outside_idx = push!(scene, medium.outside)
    mi = MediumInterfaceIdx(mat_idx, inside_idx, outside_idx)
    idx = findfirst(x -> mi === x, scene.media_interfaces)
    if idx === nothing
        @allowscalar push!(scene.media_interfaces, mi)
        idx = length(scene.media_interfaces)
    end
    return UInt32(idx)
end

function update_material!(scene::Scene, idx::UInt32, new_medium::Medium)
    mi = @allowscalar scene.media_interfaces[idx]
    Raycore.update!(scene.media, mi.inside, new_medium)
end

function update_material!(scene::Scene, idx::UInt32, new_material::Material)
    mi = @allowscalar scene.media_interfaces[idx]
    Raycore.update!(scene.materials, mi.material, new_material)
end

struct SceneHandle
    scene::Scene
    interface::UInt32 # Index into  scene.media_interfaces
    geometry::TLASHandle # handle for geometry in TLAS
end

function Base.push!(scene::Scene, mesh::AbstractGeometry, materialidx::UInt32)
    handle = push!(scene.accel, mesh, materialidx)
    return SceneHandle(scene, materialidx, handle)
end

function Base.push!(scene::Scene, mesh::AbstractGeometry, material::Material)
    mat_idx = push!(scene, material)
    return push!(scene, mesh, mat_idx)
end

# ============================================================================
# Scene operations
# ============================================================================

"""
    sync!(scene::Scene)

Build/rebuild the acceleration structure and update scene bounds.
Call this after adding geometry with `push!`.
"""
function sync!(scene::Scene{<:TLAS})
    sync!(scene.accel)
    bound = Raycore.world_bound(scene.accel)
    scene.bounds[] = (bound, bounding_sphere(bound))
    return scene
end

# Adapt Scene for GPU kernels - converts arrays to device arrays
function Adapt.adapt_structure(to, scene::Scene)
    Scene(
        Adapt.adapt(to, scene.lights),  # Convert MultiTypeSet → StaticMultiTypeSet if needed
        Adapt.adapt(to, scene.accel),
        Adapt.adapt(to, scene.materials),
        Adapt.adapt(to, scene.media),
        Adapt.adapt(to, scene.media_interfaces),
        Adapt.adapt(to, scene.bounds)  # RefValue → device RefValue
    )
end

# Type alias for scenes with materials (used for get_material dispatch)
const MaterialScene = AbstractScene

# Common interface for both scene types
@propagate_inbounds function intersect!(scene::AbstractScene, ray::AbstractRay)
    intersect!(scene.accel, ray)
end

@propagate_inbounds function intersect_p(scene::AbstractScene, ray::AbstractRay)
    hit_found, _, _, _ = any_hit(scene.accel, ray)
    return hit_found
end

# ============================================================================
# Pretty printing
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", scene::Scene)
    n_lights = length(scene.lights)
    n_material_types = length(scene.materials)
    n_materials_total = sum(length, scene.materials; init=0)
    accel = scene.accel
    accel_type = nameof(typeof(accel))

    println(io, "Scene ($accel_type):")
    println(io, "  Lights:     ", n_lights)
    if n_lights > 0
        for (i, light) in enumerate(scene.lights)
            println(io, "    [", i, "] ", typeof(light).name.name)
        end
    end
    println(io, "  Materials:  ", n_materials_total, " ($n_material_types types)")
    if accel isa TLAS
        n_instances = length(accel.instances)
        n_geometries = length(accel.blas_array)
        println(io, "  Geometries: ", n_geometries)
        println(io, "  Instances:  ", n_instances)
    elseif accel isa BVH
        n_triangles = length(accel.primitives)
        println(io, "  Triangles:  ", n_triangles)
    end
    bound = world_bound(scene)
    print(io,   "  Bounds:     ", bound.p_min, " to ", bound.p_max)
end

function Base.show(io::IO, scene::Scene)
    if get(io, :compact, false)
        n_lights = length(scene.lights)
        accel_type = nameof(typeof(scene.accel))
        n_material_types = length(scene.materials)
        print(io, "Scene($accel_type, lights=$n_lights, materials=$n_material_types)")
    else
        show(io, MIME("text/plain"), scene)
    end
end
