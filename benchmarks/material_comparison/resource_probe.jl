module ResourceProbe
using Hikari, Mantle, JSON3
using ..MaterialComparison
const Probe=MaterialComparison

function run(output; hw_accel=true)
    Mantle.enable_pipeline_executable_properties!()
    Mantle.defaultdevice!("RTX 4000 Ada")
    spec=Probe.configuration()
    records=Any[]
    mkpath(output)
    for name in ("matte","layered_principled"), implementation in (:typed_cycles,:current_cycles)
        row=only(filter(r->r.name==name,spec.cases))
        f=Probe.fixture(Probe.principled_material(row),spec; hw_accel)
        try
            Base.invokelatest(Probe.first_draw,f,implementation)
            seen=Set{UInt}()
            for pl in Hikari.allplans(f.vp.state.plans), pp in pl.passes
                pp.pass.name=="shade" || continue
                for (d,original) in zip(pp.dispatches,pp.pass.dispatches)
                    launch=d.launch
                    identity=objectid(launch.pipeline)
                    identity in seen && continue
                    push!(seen,identity)
                    label="$(name)_$(implementation)_$(length(seen))"
                    write(joinpath(output,label*".spv"),launch.compiled.spirv_bytes)
                    stats=Mantle.pipeline_exec_stats(f.dev,launch.pipeline)
                    record=(;material=name,implementation=string(implementation),hw_accel,
                             queue_type=string(typeof(first(original.args))),
                             spirv_bytes=length(launch.compiled.spirv_bytes),native=stats)
                    push!(records,record)
                    println(label," / ",stats)
                    flush(stdout)
                end
            end
            open(io->JSON3.write(io,records),joinpath(output,"resources.json"),"w")
        finally
            Probe.release(f)
        end
    end
    nothing
end
end
