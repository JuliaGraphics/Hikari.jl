module AccumulationProbe
using Hikari, Mantle, KernelAbstractions, GeometryBasics, JSON3, Statistics, Test
using ..MaterialComparison
const Probe = MaterialComparison

# A pixel has one owner. Preserve its sample order in registers, then publish
# one RGB/weight update rather than four atomics for each lane.
@kernel inbounds=true function accumulate_owned!(pixel_rgb,pixel_weight_sum,@Const(pixel_L),
        @Const(wavelengths),@Const(pdfs),@Const(weights),@Const(cie_x),@Const(cie_y),@Const(cie_z),
        @Const(n_pixels::Int32),@Const(lanes::Int32),@Const(max_component::Float32),
        @Const(output_matrix::Hikari.Mat3f),@Const(imaging_ratio::Float32))
    i = @index(Global)
    if i <= n_pixels
        table = Hikari.CIEXYZTable(cie_x,cie_y,cie_z)
        rgb_base = (i-Int32(1))*Int32(3)
        r = pixel_rgb[rgb_base+Int32(1)]
        g = pixel_rgb[rgb_base+Int32(2)]
        b = pixel_rgb[rgb_base+Int32(3)]
        w = pixel_weight_sum[i]
        for lane in Int32(0):lanes-Int32(1)
            slot = i+lane*n_pixels
            base = (slot-Int32(1))*Int32(4)
            L = Hikari.load(pixel_L,base+Int32(1),Hikari.SpectralRadiance)
            lambda = Hikari.Wavelengths(Hikari.load(wavelengths,base+Int32(1),NTuple{4,Float32}),
                                       Hikari.load(pdfs,base+Int32(1),NTuple{4,Float32}))
            response = Hikari.spectral_to_xyz(table,L,lambda)
            sensor_rgb = max.(0f0,response)*imaging_ratio
            m = maximum(sensor_rgb)
            if m > max_component
                sensor_rgb = sensor_rgb*(max_component/m)
            end
            rgb = output_matrix*sensor_rgb
            weight = weights[slot]
            weighted = weight*Vec3f(rgb[1],rgb[2],rgb[3])
            r += weighted[1]; g += weighted[2]; b += weighted[3]; w += weight
        end
        pixel_rgb[rgb_base+Int32(1)] = r
        pixel_rgb[rgb_base+Int32(2)] = g
        pixel_rgb[rgb_base+Int32(3)] = b
        pixel_weight_sum[i] = w
    end
end

function add!(g,state,refs)
    n = Int(state.width)*Int(state.height)
    Mantle.dispatch!(g,accumulate_owned!,
        (state.pixel_rgb,state.pixel_weight_sum,state.pixel_L,state.wavelengths_per_pixel,
         state.pdf_per_pixel,state.filter_weight_per_pixel,state.cie_table.cie_x,
         state.cie_table.cie_y,state.cie_table.cie_z,Int32(n),state.sobol_rng.lanes,
         refs.max_component_value,state.output_matrix,state.imaging_ratio),n;name="accumulate")
    g
end

function run(output;device,repeats=31,names=nothing,profile=false)
    Mantle.defaultdevice!(device)
    original = Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.accumulate_passes!)))
    original isa Expr || error("Cannot recover accumulation graph for restoration")
    spec = Probe.configuration()
    records = Any[]
    for row in spec.cases
        names===nothing || row.name in names || continue
        baseline = Probe.fixture(Probe.principled_material(row),spec)
        candidate = nothing
        try
            Base.invokelatest(Probe.draw,baseline)
            expected = Probe.pixels(baseline)
            Core.eval(Hikari,quote
                accumulate_passes!(g,state::VolPathState,@nospecialize(refs)) = $add!(g,state,refs)
            end)
            candidate = Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec)
            first_seconds = @elapsed Base.invokelatest(Probe.draw,candidate)
            actual = Probe.pixels(candidate)
            @test all(isfinite,actual)
            @test actual≈expected rtol=5f-5 atol=2f-5
            times = (Float64[],Float64[])
            for _ in 1:5,f in (baseline,candidate)
                Base.invokelatest(Probe.draw,f)
            end
            for i in 1:repeats,k in (isodd(i) ? (1,2) : (2,1))
                push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline : candidate))
            end
            @test Probe.pixels(candidate)==actual
            gpu = Any[]
            if profile
                for (label,f) in zip(("atomic","owned"),(baseline,candidate))
                    plans = Hikari.allplans(f.vp.state.plans)
                    for p in plans
                        p.profiler===nothing && error("No GPU timestamp profiler")
                        p.profile=true
                        Mantle.invalidate!(p)
                        foreach(empty!,p.profiler.gpu_ns)
                    end
                    for _ in 1:8
                        Base.invokelatest(Probe.draw,f)
                    end
                    passes = [t for p in plans for t in Mantle.timings(p) if t.name=="accumulate"]
                    launches = [d.launch for p in plans for pp in p.passes if pp.pass.name=="accumulate" for d in pp.dispatches]
                    @test length(launches)==1
                    mkpath(dirname(output))
                    write(joinpath(dirname(output),"$(row.name)_$label.spv"),only(launches).compiled.spirv_bytes)
                    push!(gpu,(;label,passes))
                end
            end
            result = (;name=row.name,first_seconds,lanes=Int(candidate.vp.state.sobol_rng.lanes),
                       frame_seconds=times,median_seconds=map(median,times),max_pixel_error=maximum(abs.(actual.-expected)),gpu)
            push!(records,result)
            mkpath(dirname(output));open(io->JSON3.write(io,records),output,"w")
            println(row.name," atomic/owned accumulation ms: ",result.median_seconds.*1000,
                    "; maximum pixel error ",result.max_pixel_error)
            flush(stdout)
        finally
            Core.eval(Hikari,original)
            candidate===nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    nothing
end
end
