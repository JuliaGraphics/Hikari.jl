# GPU Compatibility Tests for VolPath Kernel Inner Functions
#
# Tests kernel inner functions for GPU compatibility issues using a custom JET analyzer.
# Detects: runtime dispatch, throw statements, heap allocations.

using JET.JETInterface
using JET: JET, CC
using KernelAbstractions

# ============================================================================
# GPU Compatibility Analyzer
# ============================================================================

struct GPUAnalyzer{T} <: AbstractAnalyzer
    state::AnalyzerState
    analysis_token::AnalysisToken
    frame_filter::T
    check_dispatch::Bool
    check_throw::Bool
    check_alloc::Bool
    ignore_intrinsic_throws::Bool
    __analyze_frame::BitVector
end

JETInterface.AnalyzerState(analyzer::GPUAnalyzer) = analyzer.state
JETInterface.AbstractAnalyzer(analyzer::GPUAnalyzer, state::AnalyzerState) =
    GPUAnalyzer(state, analyzer.analysis_token, analyzer.frame_filter,
                analyzer.check_dispatch, analyzer.check_throw, analyzer.check_alloc,
                analyzer.ignore_intrinsic_throws, BitVector())
JETInterface.AnalysisToken(analyzer::GPUAnalyzer) = analyzer.analysis_token

@jetreport struct GPURuntimeDispatchReport <: InferenceErrorReport end
JETInterface.print_report_message(io::IO, ::GPURuntimeDispatchReport) =
    print(io, "runtime dispatch detected (GPU incompatible)")

@jetreport struct GPUThrowStatementReport <: InferenceErrorReport end
JETInterface.print_report_message(io::IO, ::GPUThrowStatementReport) =
    print(io, "throw statement detected (GPU incompatible)")

@jetreport struct GPUAllocationReport <: InferenceErrorReport
    alloc_type::String
end
JETInterface.print_report_message(io::IO, r::GPUAllocationReport) =
    print(io, "heap allocation detected: ", r.alloc_type, " (GPU incompatible)")

@jetreport struct GPUOptFailureReport <: InferenceErrorReport end
JETInterface.print_report_message(io::IO, ::GPUOptFailureReport) =
    print(io, "optimization failed (GPU incompatible)")

# GPU intrinsic modules - throws from these are dead code on GPU
const GPU_INTRINSIC_MODULES = Set([Base.Math])

function CC.finish!(analyzer::GPUAnalyzer, frame::CC.InferenceState, validation_world::UInt, time_before::UInt64)
    caller = frame.result
    src = caller.src
    if src isa CC.OptimizationState
        caller.src = CC.ir_to_codeinf!(src)
        frame.edges = collect(Any, caller.src.edges)
    end

    if analyzer.frame_filter(frame.linfo)
        if isa(src, Core.Const)
        elseif isnothing(src)
            add_new_report!(analyzer, caller, GPUOptFailureReport(caller.linfo))
        elseif isa(src, CC.OptimizationState)
            analyzer.check_dispatch && gpu_report_runtime_dispatch!(analyzer, caller, src)
            analyzer.check_throw && gpu_report_throw_statements!(analyzer, caller, src)
            analyzer.check_alloc && gpu_report_allocations!(analyzer, caller, src)
        end
    end

    return @invoke CC.finish!(analyzer::AbstractAnalyzer, frame::CC.InferenceState, validation_world::UInt, time_before::UInt64)
end

function gpu_report_runtime_dispatch!(analyzer::GPUAnalyzer, caller::CC.InferenceResult, opt::CC.OptimizationState)
    (; sptypes, slottypes) = opt
    for (pc, x) in enumerate(opt.src.code)
        if Base.Meta.isexpr(x, :call)
            ft = CC.widenconst(CC.argextype(first(x.args), opt.src, sptypes, slottypes))
            ft <: Core.Builtin && continue
            add_new_report!(analyzer, caller, GPURuntimeDispatchReport((opt, pc)))
        end
    end
end

function gpu_report_throw_statements!(analyzer::GPUAnalyzer, caller::CC.InferenceResult, opt::CC.OptimizationState)
    (; sptypes, slottypes) = opt
    mi = caller.linfo
    def = mi.def
    caller_module = def isa Method ? def.module : def

    for (pc, x) in enumerate(opt.src.code)
        is_throw = Base.Meta.isexpr(x, :throw)
        if !is_throw && (Base.Meta.isexpr(x, :invoke) || Base.Meta.isexpr(x, :call))
            args = x.args
            if length(args) >= 1
                ft = Base.Meta.isexpr(x, :invoke) && length(args) >= 2 ?
                    CC.argextype(args[2], opt.src, sptypes, slottypes) :
                    CC.argextype(first(args), opt.src, sptypes, slottypes)
                if ft !== nothing
                    f = CC.singleton_type(ft)
                    if f in (Base.throw, Core.throw, Base.error, Base.ArgumentError,
                             Base.BoundsError, Base.DomainError)
                        is_throw = true
                    end
                end
            end
        end
        if is_throw
            analyzer.ignore_intrinsic_throws && caller_module in GPU_INTRINSIC_MODULES && continue
            add_new_report!(analyzer, caller, GPUThrowStatementReport((opt, pc)))
        end
    end
end

function gpu_report_allocations!(analyzer::GPUAnalyzer, caller::CC.InferenceResult, opt::CC.OptimizationState)
    (; sptypes, slottypes) = opt
    for (pc, x) in enumerate(opt.src.code)
        alloc_type = nothing
        if Base.Meta.isexpr(x, :new) && length(x.args) >= 1
            T = CC.widenconst(CC.argextype(x.args[1], opt.src, sptypes, slottypes))
            if T isa DataType && Base.ismutabletype(T)
                alloc_type = "mutable struct $(T.name.name)"
            end
        elseif Base.Meta.isexpr(x, :call) && length(x.args) >= 1
            first_arg = x.args[1]
            if first_arg isa GlobalRef && first_arg.mod === Core && first_arg.name === :memorynew
                alloc_type = "memory allocation"
            end
        elseif Base.Meta.isexpr(x, :foreigncall)
            fname = x.args[1]
            fname isa QuoteNode && (fname = fname.value)
            if fname isa Symbol
                s = String(fname)
                if startswith(s, "jl_alloc") || s in ("jl_array_copy", "jl_new_array")
                    alloc_type = "array ($s)"
                end
            end
        end
        alloc_type !== nothing && add_new_report!(analyzer, caller, GPUAllocationReport((opt, pc), alloc_type))
    end
end

const _gpu_analysis_token = AnalysisToken()

function GPUAnalyzer(world::UInt = Base.get_world_counter();
    analysis_token::AnalysisToken = _gpu_analysis_token,
    frame_filter = (_::Core.MethodInstance) -> true,
    check_dispatch::Bool = true,
    check_throw::Bool = true,
    check_alloc::Bool = true,
    ignore_intrinsic_throws::Bool = true,
    jetconfigs...)
    state = AnalyzerState(world; jetconfigs...)
    GPUAnalyzer(state, analysis_token, frame_filter, check_dispatch, check_throw, check_alloc, ignore_intrinsic_throws, BitVector())
end

function report_gpu_compat(args...; jetconfigs...)
    analyzer = GPUAnalyzer(; jetconfigs...)
    JET.analyze_and_report_call!(analyzer, args...; jetconfigs...)
end

gpu_module_filter(m::Module) = (mi::Core.MethodInstance) -> begin
    def = mi.def
    (def isa Method ? def.module : def) === m
end

"""
Test that a function has no runtime dispatch (GPU compatible).
Returns true if no dispatch issues found.
"""
function test_no_dispatch(f, types; frame_filter=gpu_module_filter(Hikari))
    result = report_gpu_compat(f, types; frame_filter, check_throw=false, check_alloc=false)
    reports = JET.get_reports(result)
    dispatch_count = count(r -> r isa GPURuntimeDispatchReport, reports)
    dispatch_count == 0
end

# ============================================================================
# Test Scene Generator
# ============================================================================

function gen_gpu_test_scene()
    # Materials
    white_matte = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.73f0))
    glass = Hikari.GlassMaterial(index=1.5f0)
    emissive = Hikari.MediumInterface(Hikari.EmissiveMaterial(Le=Hikari.RGBSpectrum(10f0)))

    # Media
    fog = Hikari.HomogeneousMedium(
        σ_a=Hikari.RGBSpectrum(0.01f0),
        σ_s=Hikari.RGBSpectrum(0.1f0),
        Le=Hikari.RGBSpectrum(0f0),
        g=0.3f0
    )
    glass_with_fog = Hikari.MediumInterface(glass; inside=fog, outside=nothing)

    # Geometry
    function tmesh(prim, material)
        prim_tess = prim isa Sphere ? Tesselation(prim, 16) : prim
        mesh = normal_mesh(prim_tess)
        Hikari.GeometricPrimitive(Raycore.TriangleMesh(mesh), material)
    end

    primitives = [
        tmesh(Rect3f(Vec3f(-1, 0, -1), Vec3f(2, 0.01f0, 2)), white_matte),
        tmesh(Sphere(Point3f(0, 0.5f0, 0), 0.3f0), glass_with_fog),
        tmesh(Sphere(Point3f(0, 1.5f0, 0), 0.1f0), emissive),
    ]
    mat_scene = Hikari.MaterialScene(primitives)

    # Lights
    lights = (
        Hikari.PointLight(Point3f(0f0, 2f0, 0f0), Hikari.RGBSpectrum(10f0)),
        Hikari.AmbientLight(Hikari.RGBSpectrum(0.1f0)),
    )

    scene = Hikari.Scene(lights, mat_scene)

    # VolPathState for rgb2spec_table
    state = Hikari.VolPathState(KernelAbstractions.CPU(), 64, Int32(4))

    (
        scene=scene,
        mat_scene=mat_scene,
        materials=mat_scene.materials,
        media=mat_scene.media,
        textures=mat_scene.textures,
        lights=lights,
        rgb2spec_table=state.rgb2spec_table,
        lambda=Hikari.SampledWavelengths{4}((400f0, 500f0, 600f0, 700f0), (1f0, 1f0, 1f0, 1f0)),
        mat_idx=Hikari.SetKey(UInt8(1), UInt32(1)),
        ray=Raycore.Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=Inf32),
    )
end

# ============================================================================
# Tests
# ============================================================================

@testset "GPU Compatibility: VolPath Kernel Functions" begin
    td = gen_gpu_test_scene()

    @testset "Material Dispatch Functions" begin
        @test test_no_dispatch(Hikari.is_emissive_dispatch, (typeof(td.materials), typeof(td.mat_idx)))
        @test test_no_dispatch(Hikari.is_pure_emissive_dispatch, (typeof(td.materials), typeof(td.mat_idx)))
        @test test_no_dispatch(Hikari.has_bsdf_dispatch, (typeof(td.materials), typeof(td.mat_idx)))
        @test test_no_dispatch(Hikari.has_medium_interface_dispatch, (typeof(td.materials), typeof(td.mat_idx)))
    end

    @testset "Medium Dispatch Functions" begin
        @test test_no_dispatch(Hikari.sample_point_dispatch,
            (typeof(td.rgb2spec_table), typeof(td.media), Int32, Point3f, typeof(td.lambda)))
        @test test_no_dispatch(Hikari.get_majorant_dispatch,
            (typeof(td.rgb2spec_table), typeof(td.media), Int32, typeof(td.ray), Float32, Float32, typeof(td.lambda)))
    end

    @testset "Phase Functions" begin
        @test test_no_dispatch(Hikari.hg_p, (Float32, Float32))
        @test test_no_dispatch(Hikari.sample_hg, (Float32, Vec3f, Point2f))
    end

    @testset "Light Sampling" begin
        @test test_no_dispatch(Hikari.sample_light_from_tuple,
            (typeof(td.rgb2spec_table), typeof(td.lights), Int32, Point3f, typeof(td.lambda), Point2f))
    end

    @testset "Geometry Sampling" begin
        @test test_no_dispatch(Hikari.concentric_sample_disk, (Point2f,))
        @test test_no_dispatch(Hikari.cosine_sample_hemisphere, (Point2f,))
        @test test_no_dispatch(Hikari.uniform_sample_sphere, (Point2f,))
        @test test_no_dispatch(Hikari.uniform_sample_cone, (Point2f, Float32))
    end

    @testset "Wavelength Sampling" begin
        @test test_no_dispatch(Hikari.sample_visible_wavelengths, (Float32,))
    end

    @testset "BSDF Sampling" begin
        matte_mat = first(td.materials[1])
        glass_mat = first(td.materials[2])
        emissive_mat = first(td.materials[3])

        bsdf_types = (typeof(td.rgb2spec_table), Any, typeof(td.textures), Vec3f, Vec3f, Point2f, typeof(td.lambda), Point2f, Float32)

        @test test_no_dispatch(Hikari.sample_bsdf_spectral,
            (typeof(td.rgb2spec_table), typeof(matte_mat), typeof(td.textures), Vec3f, Vec3f, Point2f, typeof(td.lambda), Point2f, Float32))
        @test test_no_dispatch(Hikari.sample_bsdf_spectral,
            (typeof(td.rgb2spec_table), typeof(glass_mat), typeof(td.textures), Vec3f, Vec3f, Point2f, typeof(td.lambda), Point2f, Float32))
        @test test_no_dispatch(Hikari.sample_bsdf_spectral,
            (typeof(td.rgb2spec_table), typeof(emissive_mat), typeof(td.textures), Vec3f, Vec3f, Point2f, typeof(td.lambda), Point2f, Float32))
    end

    @testset "Emission Functions" begin
        @test test_no_dispatch(Hikari.get_emission_spectral_dispatch,
            (typeof(td.rgb2spec_table), typeof(td.materials), typeof(td.textures), typeof(td.mat_idx), Vec3f, Vec3f, Point2f, typeof(td.lambda)))
        @test test_no_dispatch(Hikari.get_emission_spectral_uv_dispatch,
            (typeof(td.rgb2spec_table), typeof(td.materials), typeof(td.textures), typeof(td.mat_idx), Point2f, typeof(td.lambda)))
    end

    @testset "Microfacet/Fresnel Functions" begin
        @test test_no_dispatch(Hikari.sample_dielectric_transmission_spectral, (Float32, Vec3f, Float32))
        @test test_no_dispatch(Hikari.sample_ggx_vndf, (Vec3f, Float32, Float32, Point2f))
        @test test_no_dispatch(Hikari.trowbridge_reitz_sample, (Vec3f, Float32, Float32, Float32, Float32))
    end
end
