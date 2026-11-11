module MaterialComparison
using Hikari, Mantle, GeometryBasics, JSON3, Statistics, Test
using SHA: sha256
using GeometryBasics: Point2f, Point3f, Sphere, Tesselation, normal_mesh

configuration(path = joinpath(@__DIR__, "cases.json")) = JSON3.read(read(path, String))

# Include untracked source files too: a Git diff alone misses new implementations.
function source_hash(mod::Module = Hikari)
    root = joinpath(pkgdir(mod), "src")
    files = sort!([joinpath(dir, file) for (dir, _, names) in walkdir(root)
                   for file in names if endswith(file, ".jl")])
    bytes = IOBuffer()
    for file in files
        write(bytes, relpath(file, root), '\0', read(file), '\0')
    end
    return bytes2hex(sha256(take!(bytes)))
end

function original_material(name)
    name == "diffuse" && return Hikari.Diffuse(Kd = (0.6f0, 0.6f0, 0.6f0))
    name == "dielectric" && return Hikari.Dielectric(roughness = 0.04f0, index = 1.5f0, remap_roughness = false)
    name == "conductor" && return Hikari.Conductor(eta = (0.2f0, 0.2f0, 0.2f0), k = (3.9f0, 3.9f0, 3.9f0),
                                                  roughness = 0.04f0, remap_roughness = false)
    name == "coated_diffuse" && return Hikari.CoatedDiffuse(reflectance = (0.5f0, 0.5f0, 0.5f0),
                                                           roughness = 0.04f0, eta = 1.5f0,
                                                           remap_roughness = false)
    error("Unknown original material: $name")
end

function principled_material(row, spec = configuration())
    haskey(row, :materials) && return map(r -> principled_material(r, spec), row.materials)
    if haskey(row, :mix)
        return Hikari.Mix(principled_material(row.mix.a, spec), principled_material(row.mix.b, spec);
                          factor = Float32(row.mix.factor))
    end
    inputs = (; (Symbol(k) => (v isa AbstractVector ? Tuple(Float32.(v)) : Float32(v))
                 for (k, v) in pairs(row.principled))...)
    return Hikari.Principled(; inputs..., distribution = Symbol(spec.distribution))
end

add_materials!(scene, mesh, material, spec) = push!(scene, mesh, material)
function add_materials!(scene, mesh, materials::AbstractVector, spec)
    length(materials) == length(spec.instances) || error("One material is required per instance")
    return map(materials, spec.instances) do material, position
        transform = Hikari.Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                Float32(position[1]), Float32(position[2]), Float32(position[3]), 1)
        push!(scene, mesh, material; transform)
    end
end

function fixture(material, spec; samples = spec.samples, resolution = spec.resolution, hw_accel = true, rr_depth = 3)
    scene = Hikari.Scene(; backend = Mantle.defaultbackend(), hw_accel)
    mesh = normal_mesh(Tesselation(Sphere(Point3f(0), Float32(spec.sphere_radius)), 32))
    add_materials!(scene, mesh, material, spec)
    light = spec.point_light
    push!(scene, Hikari.PointLight(Point3f(light.position...), Hikari.RGBSpectrum(Float32(light.intensity))))
    environment = Float32(get(spec, :environment_strength, 0))
    environment > 0 && push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(environment)))
    Hikari.sync!(scene)
    film = Hikari.Film(Mantle.defaultbackend(), Hikari.Film(Point2f(resolution, resolution)))
    camera = Hikari.PerspectiveCamera(Point3f(spec.camera.position...), Point3f(spec.camera.lookat...),
                                      film; fov = Float32(spec.camera.fov))
    vp = Hikari.VolPath(; samples, max_depth = Int(spec.max_depth), russian_roulette_depth = rr_depth,
                        regularize = false, max_component_value = Inf32, filter = Hikari.BoxFilter(),
                        hw_accel)
    return (; scene, film, camera, vp, mesh, dev = Hikari.mantle_device(Mantle.defaultbackend()))
end

draw(f) = (f.vp(f.scene, f.film, f.camera); Mantle.waitidle(f.dev); nothing)
pixels(f) = Float32[x for p in Array(f.film.framebuffer) for x in (p.r, p.g, p.b)]
release(f) = (close(f.vp); Hikari.free!(f.film); nothing)

function shade_signatures(f)
    return unique([string(Mantle.devicetype(f.dev, d.args))
                   for plan in Hikari.allplans(f.vp.state.plans) for pass in plan.passes
                   if startswith(pass.pass.name, "shade") for d in pass.pass.dispatches])
end

function first_draw(f, implementation)
    implementation != :typed_cycles && return draw(f)
    @eval Hikari.bsdf_kernel_nodes(::Type{M}, ::Mantle.Backend) where {M <: Tuple} = M
    method = which(Hikari.bsdf_kernel_nodes, Tuple{Type{Tuple{}}, Mantle.VulkanAPI})
    try
        return Base.invokelatest(draw, f)
    finally
        Base.delete_method(method)
    end
end

function write_obj(path, mesh)
    open(path, "w") do io
        for p in coordinates(mesh)
            println(io, "v ", join(p, ' '))
        end
        for n in mesh.normal
            println(io, "vn ", join(n, ' '))
        end
        for face in faces(mesh)
            println(io, "f ", join(("$j//$j" for j in Int.(GeometryBasics.value.(face))), ' '))
        end
    end
end

function run(output; original = false, samples = nothing, resolution = nothing, repeats = nothing,
             only_case = nothing, only_implementation = nothing, select = "RTX 4000 Ada",
             spec = configuration(), save_images = true)
    mkpath(output)
    samples === nothing && (samples = Int(spec.samples))
    resolution === nothing && (resolution = Int(spec.resolution))
    repeats === nothing && (repeats = Int(spec.repeats))
    device = Mantle.defaultdevice!(select)
    name = Mantle.devicename(device)
    result = Dict{String,Any}("device" => name, "julia" => string(VERSION),
                             "hikari_commit" => strip(read(Cmd(["git", "-C", pkgdir(Hikari), "rev-parse", "HEAD"]), String)),
                             "hikari_source_diff_sha256" => bytes2hex(sha256(read(Cmd(["git", "-C", pkgdir(Hikari), "diff", "--", "src"])))),
                             "hikari_source_sha256" => source_hash(),
                             "resolution" => resolution, "samples" => samples,
                             "distribution" => spec.distribution,
                             "environment_strength" => get(spec, :environment_strength, 0),
                             "max_depth" => spec.max_depth, "rows" => Any[])
    outfile = joinpath(output, original ? "original_hikari.json" : "hikari_cycles.json")
    for row in spec.cases
        only_case !== nothing && row.name != only_case && continue
        original && row.original === nothing && continue
        implementations = only_implementation !== nothing ? (only_implementation,) :
                          original ? (:original_hikari,) : (:typed_cycles, :current_cycles)
        fixtures = Any[]
        records = Any[]
        images = Vector{Float32}[]
        try
            for impl in implementations
                println("Constructing ", row.name, " / ", impl); flush(stdout)
                material = original ? original_material(row.original) : principled_material(row, spec)
                Mantle.Lava.reset_compile_stats!()
                setup = @elapsed f = fixture(material, spec; samples, resolution)
                push!(fixtures, f)
                startup = @elapsed Base.invokelatest(first_draw, f, impl)
                image = pixels(f)
                push!(images, image)
                @test all(isfinite, image)
                @test maximum(image) > 0
                signatures = shade_signatures(f)
                if !original
                    @test any(s -> occursin("SurfaceProgramNodes", s), signatures) == (impl == :current_cycles)
                end
                if save_images
                    write_obj(joinpath(output, "sphere.obj"), f.mesh)
                    open(io -> write(io, reinterpret(UInt8, image)), joinpath(output, "$(row.name)_$impl.f32"), "w")
                end
                push!(records, Dict("implementation" => string(impl), "setup_seconds" => setup,
                                   "first_render_seconds_existing_caches" => startup,
                                   "spirv_cache" => Mantle.Lava.compile_stats(),
                                   "llvm_cache" => Mantle.Lava.llvm_compile_stats(),
                                   "function_cache" => Mantle.Lava.function_compile_stats(),
                                   "shade_signatures" => signatures, "frame_seconds" => Float64[],
                                   "image_mean" => mean(image), "image_max" => maximum(image)))
                records[end]["image_sha256"] = bytes2hex(sha256(reinterpret(UInt8, image)))
                if !original
                    records[end]["wave_lanes"] = Int(f.vp.state.sobol_rng.lanes)
                    records[end]["wave_paths"] = Int(f.vp.state.ray_queue_a.capacity)
                    records[end]["device_budget_bytes"] = Mantle.capacity(f.dev)
                    records[end]["shade_workgroups"] = unique([
                        d.launch.compiled.workgroup_size for plan in Hikari.allplans(f.vp.state.plans)
                        for pass in plan.passes if pass.pass.name == "shade" for d in pass.dispatches])
                end
                println("First render: ", startup, " seconds"); flush(stdout)
            end
            if !original && length(fixtures) == 2
                a, b = pixels.(fixtures)
                @test a ≈ b rtol = 5f-5 atol = 2f-5
                records[2]["max_error_against_typed_cycles"] = maximum(abs.(a .- b))
            end
            for _ in 1:Int(spec.warmups), f in fixtures
                Base.invokelatest(draw, f)
            end
            for (k,f) in enumerate(fixtures)
                warmed = pixels(f)
                @test warmed ≈ images[k] rtol = 5f-5 atol = 2f-5
                records[k]["max_error_after_warmup"] = maximum(abs.(warmed .- images[k]))
            end
            for i in 1:repeats, k in (isodd(i) ? eachindex(fixtures) : reverse(eachindex(fixtures)))
                seconds = @elapsed Base.invokelatest(draw, fixtures[k])
                push!(records[k]["frame_seconds"], seconds)
            end
            for record in records
                record["median_seconds"] = median(record["frame_seconds"])
                println(row.name, " / ", record["implementation"], ": ", record["median_seconds"] * 1000, " ms")
            end
            for (k,f) in enumerate(fixtures)
                final = pixels(f)
                @test final ≈ images[k] rtol = 5f-5 atol = 2f-5
                records[k]["max_error_after_timing"] = maximum(abs.(final .- images[k]))
            end
            push!(result["rows"], Dict("material" => String(row.name), "variants" => records))
            open(io -> JSON3.write(io, result), outfile, "w")
            flush(stdout)
        finally
            foreach(release, fixtures)
        end
    end
    return nothing
end
end
