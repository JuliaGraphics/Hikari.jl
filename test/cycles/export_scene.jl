# The scenes of `material_cases.jl` for Blender: each batch's meshes as OBJ and the rest as JSON, which
# `cycles_render.py` builds and renders with Cycles. How the references in `test_cycles_materials.jl` were
# made (Blender 5.2.1, OptiX, 4096 samples):
#
#     include("test/cycles/export_scene.jl"); export_cycles_scenes("out/cycles"; samples = 4096)
#     blender -b --python-exit-code 1 -P test/cycles/cycles_render.py -- out/cycles/b1/scene.json out/cycles/b1/cycles.exr OPTIX 1
#
# and the means of each sphere taken with `cycles_sphere_pixels` and `cycles_mean`.
include(joinpath(@__DIR__, "material_cases.jl"))
using JSON

function write_obj(path, m)
    open(path, "w") do io
        for p in coordinates(m); println(io, "v ", p[1], " ", p[2], " ", p[3]); end
        for n in m.normal; println(io, "vn ", n[1], " ", n[2], " ", n[3]); end
        for f in faces(m)
            println(io, "f ", join(("$j//$j" for j in Int.(GeometryBasics.value.(f))), " "))
        end
    end
end

function export_cycles_scenes(dir; samples, batches = cycles_batches())
    for (b, cases) in enumerate(batches)
        out = mkpath(joinpath(dir, "b$b"))
        objects = Any[]
        for ((name, (_, blender)), c) in zip(cases, CYCLES_CENTERS)
            path = joinpath(out, "sphere_$name.obj"); write_obj(path, cycles_sphere(c))
            push!(objects, Dict("obj" => path, "material" => blender))
        end
        write_obj(joinpath(out, "ground.obj"), cycles_ground())
        push!(objects, Dict("obj" => joinpath(out, "ground.obj"), "material" => Dict("type" => "diffuse", "color" => [0.5, 0.5, 0.5])))
        write_obj(joinpath(out, "light.obj"), cycles_light())
        push!(objects, Dict("obj" => joinpath(out, "light.obj"),
                            "material" => Dict("type" => "emission", "strength" => CYCLES_SCENE.light.strength)))
        cam = CYCLES_SCENE.camera
        spec = Dict("resolution" => CYCLES_SCENE.res, "samples" => samples, "max_bounces" => CYCLES_SCENE.max_depth,
                    "world" => CYCLES_SCENE.world,
                    "camera" => Dict("position" => collect(cam.position), "lookat" => collect(cam.lookat),
                                     "up" => collect(cam.up), "fov" => cam.fov),
                    "objects" => objects)
        open(io -> JSON.print(io, spec, 2), joinpath(out, "scene.json"), "w")
    end
end
