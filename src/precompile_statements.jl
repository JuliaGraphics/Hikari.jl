_precompile_device_types() = Mantle._precompile_device_types()

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
                                :sampler_seed, :accumulation_eltype, :sensor, :has_media),
                               Tuple{Int32, Int32, Int, Float32, Int, UInt32, DataType, PixelSensor, Bool}}
    render_kwargs = NamedTuple{(:samples, :finalize_framebuffer), Tuple{Int, Bool}}
    for device in _precompile_device_types()
        films = Base.uniontypes(Core.Compiler.return_type(Film, Tuple{device, typeof(film)}))
        flags = NamedTuple{(:has_media, :has_lights, :emitters_reachable, :crossings), Tuple{Bool, Bool, Bool, Bool}}
        for gpu_film in films
            isconcretetype(gpu_film) && gpu_film <: Film || continue
            for camera in (PerspectiveCamera, MatrixCamera)
                push!(signatures, Tuple{typeof(Core.kwcall), flags, typeof(build_plans), device,
                                        VolPathState{device}, fieldtype(gpu_film, :framebuffer), Any, PerRun{camera}, Int32})
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
