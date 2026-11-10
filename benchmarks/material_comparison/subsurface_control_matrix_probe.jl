module SubsurfaceControlMatrixProbe
using Hikari, Mantle, Revise, JSON3
using ..SubgroupControlProbe

function run(output; device)
    Mantle.defaultdevice!(device)
    original = Revise.CodeTracking.definition(Expr, only(methods(Hikari.shader_bsdf_workgroupsize)))
    original isa Expr || error("Cannot recover shade workgroup sizing")
    cases = [p for batch in Main.cycles_batches() for p in batch
             if p[1] in ("subsurface_node_zero_channel", "mix_subsurface_diffuse")]
    records = Any[]
    L = Mantle.Lava
    world = L._initialization_world[]
    try
        for (width, group) in ((0, 64), (0, 32), (32, 64), (32, 32))
            Mantle.clear_kernel_cache!(device)
            Core.eval(Hikari, :(shader_bsdf_workgroupsize(group) = min(group, $group)))
            L._initialization_world[] = Base.get_world_counter()
            function measure()
                image = Base.invokelatest(Main.render_cycles_batch, cases; samples = 64)
                means = [Main.cycles_mean(image, Main.cycles_sphere_pixels(k)) for k in eachindex(cases)]
                walk = Hikari.Principled(base_color=(1,1,1), roughness=.4, subsurface_weight=1,
                    subsurface_radius=(1,1,1), subsurface_scale=.1, ior=1.45)
                furnace = Base.invokelatest(Main.furnace_mean, walk; regularize=false, res=32, spp=64)
                push!(records, (; width, group, means, furnace))
                open(io -> JSON3.write(io, records), output, "w")
                println("SUBSURFACE_CONTROL ", last(records)); flush(stdout)
            end
            if width == 0
                Base.invokelatest(measure)
            else
                SubgroupControlProbe.run(measure, device; width)
            end
        end
    finally
        Core.eval(Hikari, original)
        L._initialization_world[] = world
    end
    nothing
end
end
