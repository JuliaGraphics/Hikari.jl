# Resolve backend types by inference, without probing hardware or constructing
# a device. This keeps the workload portable and avoids baking one backend's
# concrete names into generated precompile statements.
function _precompile_device_types()
    result = DataType[]
    for method in methods(Mantle.makedevice)
        signature = Base.unwrap_unionall(method.sig)
        length(signature.parameters) == 4 || continue
        api = signature.parameters[2]
        api isa DataType && isconcretetype(api) && api <: Mantle.Backend || continue
        device = Core.Compiler.return_type(Mantle.makedevice, Tuple{api, Nothing, Mantle.DebugConfig})
        device <: Mantle.Device && isconcretetype(device) && push!(result, device)
    end
    return unique(result)
end

function _precompile_statements()
    mesh = normal_mesh(Tesselation(Sphere(Point3f(0), 1f0), 8))
    metadata = fill(TriangleMeta(0, 0, 0), length(faces(mesh)))
    traced_mesh = GeometryBasics.mesh(mesh; face_meta = GeometryBasics.per_face(metadata, mesh))
    film = Film(Point2f(8, 8))
    descriptions = (Diffuse(color = (0.5, 0.5, 0.5)), Principled(), Glass(), Metallic(),
                    Toon(), Toon(smooth = 0.2))
    signatures = Any[Tuple{typeof(parse_pbrt_string), String}]
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
            push!(signatures, Tuple{typeof(Core.kwcall), NamedTuple{(:opaque,), Tuple{Bool}},
                                    typeof(push!), fieldtype(scene, :accel), typeof(traced_mesh), Mat4f})
            for material in descriptions
                push!(signatures, Tuple{typeof(push!), scene, typeof(mesh), typeof(material)})
            end
        end
    end
    return count(Base.precompile, signatures), length(signatures)
end
