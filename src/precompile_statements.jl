_precompile_device_types() = Mantle._precompile_device_types()

# Describe the common opaque triangle/point-light dispatch without allocating
# arrays or initializing a device. Keep this in sync with shade_pass!; a test
# compares these signatures with those of real recorded shading dispatches.
function _precompile_shader_signatures(device::Type{<:Mantle.Device}; hw_accel::Bool = true,
                                      readonly::Bool = true)
    abi = Mantle._kernel_precompile_abi(device, 32)
    abi === nothing && return ()
    A(T, N = 1) = abi.array(T, N)
    C(T, N = 1) = readonly ? abi.readonly(T, N) : abi.array(T, N)
    context = abi.context
    fields = NamedTuple{fieldnames(VPRayWorkItem),Tuple{map(A,fieldtypes(VPRayWorkItem))...}}
    rays = StructArray{VPRayWorkItem,1,fields,Int}
    ray_queue = WorkQueue{VPRayWorkItem,rays,A(Int32)}
    read_fields = NamedTuple{fieldnames(VPRayWorkItem),Tuple{map(C,fieldtypes(VPRayWorkItem))...}}
    read_rays = StructArray{VPRayWorkItem,1,read_fields,Int}
    read_queue = WorkQueue{VPRayWorkItem,read_rays,C(Int32)}
    hit_queue = WorkQueue{VPSurfaceHit,C(VPSurfaceHit),C(Int32)}
    inputs = ShaderInputs{C(Float32),C(Int32)}
    nodes = SurfaceProgramNodes{16,true,SURFACE_PRINCIPLED_MASK | SURFACE_ONE_PRINCIPLED,false}
    queue = ClosureQueue{nodes,C(Int32),inputs}
    queues = Core.Compiler.return_type(shade_queues, Tuple{queue})
    triangle = Raycore.Triangle{TriangleMeta}
    accel = if hw_accel
        Mantle.AdaptedAccel{Nothing,A(triangle),A(UInt32),triangle,Nothing,Nothing,A(Raycore.InstanceRecord)}
    else
        Raycore.StaticTLAS{A(Raycore.BVHNode2),A(Raycore.InstanceDescriptor),
                          A(Raycore.BVHNode2),A(triangle),A(Raycore.BLASDescriptor)}
    end
    media = StaticMultiTypeSet{Tuple{},Tuple{}}
    textures = Tuple{C(A(LookupValue)),C(A(NodeInstr{NodeFeature.CLOSURES}))}
    materials = Raycore.SetView{(),(1,3),Tuple{},textures}
    lights = StaticMultiTypeSet{Tuple{C(PointLight{RGBSpectrum})},Tuple{}}
    rgb = RGBToSpectrumTable{C(Float32),C(Float32,5),C(Float32)}
    tail = (hit_queue,read_queue,read_queue,ray_queue,A(Float32),A(Float32),
        accel,C(MediumInterfaceIdx),media,materials,lights,rgb,C(LightBVHNode),
        C(Int32),A(UInt32),Int32,Int32,Int32,Int32,Bool,Bool,
        SobolRNG{C(UInt32)},C(Int32),C(PerspectiveCamera),Int32,Int32)
    return [Tuple{typeof(gpu_vp_shade_kernel!),context,Q,tail...} for Q in queues.parameters]
end

function _precompile_shader_workload(device::Type{<:Mantle.Device})
    cache = Mantle.AccessCache()
    # Access analysis runs before readonly adaptation, while code generation
    # sees the adapted signature. Both inference roots must reach the image.
    for readonly in (false, true), hw_accel in (true, false),
        sig in _precompile_shader_signatures(device; hw_accel, readonly)
        args = Tuple(sig.parameters[2:end])
        interp = Mantle._precompile_kernelinterpreter(device, gpu_vp_shade_kernel!, Tuple{args...}, 32)
        Mantle.accessof(interp, gpu_vp_shade_kernel!, args; cache)
    end
    return nothing
end

function _precompile_statements()
    mesh = normal_mesh(Tesselation(Sphere(Point3f(0), 1f0), 8))
    metadata = fill(TriangleMeta(0, 0, 0), length(faces(mesh)))
    traced_mesh = GeometryBasics.mesh(mesh; face_meta = GeometryBasics.per_face(metadata, mesh))
    film = Film(Point2f(8, 8))
    descriptions = (Diffuse(color = (0.5, 0.5, 0.5)), Principled(), Glass(), Metallic(),
                    Toon(), Toon(smooth = 0.2))
    signatures = Any[Tuple{typeof(parse_pbrt_string), String}]
    # These host initializers are independent of the scene's material types.
    state_kwargs = NamedTuple{(:max_depth, :rr_depth, :lanes, :scene_radius, :samples_per_pixel,
                                :sampler_seed, :accumulation_eltype, :accumulators, :sensor, :has_media),
                               Tuple{Int32, Int32, Int, Float32, Int, UInt32, DataType, Nothing, PixelSensor, Bool}}
    render_kwargs = NamedTuple{(:samples, :finalize_framebuffer), Tuple{Int, Bool}}
    for device in _precompile_device_types()
        films = Base.uniontypes(Core.Compiler.return_type(Film, Tuple{device, typeof(film)}))
        flags = NamedTuple{(:has_media, :has_lights, :emitters_reachable, :crossings), Tuple{Bool, Bool, Bool, Bool}}
        for gpu_film in films
            isconcretetype(gpu_film) && gpu_film <: Film || continue
            for camera in (PerspectiveCamera, MatrixCamera)
                push!(signatures, Tuple{typeof(Core.kwcall), flags, typeof(build_plans), device,
                                        VolPathState{device}, fieldtype(gpu_film, :framebuffer), Any, PerRun{camera}, Int32})
                push!(signatures, Tuple{typeof(Core.kwcall), flags, typeof(rebuild_plans!),
                    VolPathState{device}, gpu_film, device, Any, Any, Any, Any, Any,
                    camera, SetKey, GPUFilterParams, Nothing, Bool, Int32, Float32, Int32})
            end
        end
        kwargs = NamedTuple{(:backend,), Tuple{device}}
        push!(signatures, Tuple{typeof(Core.kwcall), kwargs, Type{Scene}})
        scenes = Base.uniontypes(Core.Compiler.return_type(Core.kwcall, Tuple{kwargs, Type{Scene}}))
        for scene in scenes
            isconcretetype(scene) && scene <: Scene || continue
            push!(signatures, Tuple{typeof(sync!), scene})
            # Compile the render entry points with the same concrete scene and
            # film types as the initializers above. These only infer and emit
            # host code; no scene, device, or GPU kernel is constructed here.
            for gpu_film in films
                isconcretetype(gpu_film) && gpu_film <: Film || continue
                for camera in (PerspectiveCamera, MatrixCamera)
                    push!(signatures, Tuple{VolPath, scene, gpu_film, camera})
                    push!(signatures, Tuple{typeof(Core.kwcall), render_kwargs,
                                            typeof(render!), VolPath, scene, gpu_film, camera})
                end
            end
            push!(signatures, Tuple{typeof(Core.kwcall), state_kwargs, Type{VolPathState},
                                    device, Int, Int, Raycore.MultiTypeSet{device}})
            triangles = Core.Compiler.return_type(Adapt.adapt, Tuple{device, Vector{Raycore.Triangle{TriangleMeta}}})
            isconcretetype(triangles) && triangles <: AbstractArray &&
                push!(signatures, Tuple{typeof(Raycore.build_blas), triangles})
            push!(signatures, Tuple{typeof(Core.kwcall), NamedTuple{(:opaque,), Tuple{Bool}},
                                    typeof(push!), fieldtype(scene, :accel), typeof(traced_mesh), Mat4f})
            for material in descriptions
                push!(signatures, Tuple{typeof(push!), scene, typeof(mesh), typeof(material)})
            end
        end
    end
    return count(Base.precompile, signatures), length(signatures)
end
