using JET
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

struct GPUAnalyzer{T, MT <: CC.MethodTableView} <: AbstractAnalyzer
    state::AnalyzerState
    analysis_token::AnalysisToken
    # The device's: its overlay holds the shader bodies of device intrinsics, array indexing
    # included, where the host has error stubs. On the host's, inference ends at a kernel's
    # first array read and finds nothing after it.
    method_table::MT
    frame_filter::T
    check_dispatch::Bool
    check_throw::Bool
    check_alloc::Bool
    ignore_intrinsic_throws::Bool
    __analyze_frame::BitVector
end

JETInterface.AnalyzerState(analyzer::GPUAnalyzer) = analyzer.state
JETInterface.AbstractAnalyzer(analyzer::GPUAnalyzer, state::AnalyzerState) =
    GPUAnalyzer(state, analyzer.analysis_token, analyzer.method_table, analyzer.frame_filter,
                analyzer.check_dispatch, analyzer.check_throw, analyzer.check_alloc,
                analyzer.ignore_intrinsic_throws, BitVector())
JETInterface.AnalysisToken(analyzer::GPUAnalyzer) = analyzer.analysis_token
CC.method_table(analyzer::GPUAnalyzer) = analyzer.method_table

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

# The checks read the optimized IR, before it becomes the cached CodeInfo, and report at its
# statements through JET's `OptimizedIRState`. They reported at `(OptimizationState, pc)`, which
# JET no longer takes: the first dispatch found threw a `MethodError` instead of being reported.
function CC.finish!(analyzer::GPUAnalyzer, frame::CC.InferenceState, validation_world::UInt, time_before::UInt64)
    caller = frame.result
    src = caller.src
    if analyzer.frame_filter(frame.linfo)
        if isnothing(src)
            add_new_report!(analyzer, caller, GPUOptFailureReport(caller.linfo))
        elseif src isa CC.OptimizationState
            state = JET.OptimizedIRState(src, JET.optimized_ir(src))
            analyzer.check_dispatch && gpu_report_runtime_dispatch!(analyzer, caller, state)
            analyzer.check_throw && gpu_report_throw_statements!(analyzer, caller, state)
            analyzer.check_alloc && gpu_report_allocations!(analyzer, caller, state)
        end
    end
    if src isa CC.OptimizationState
        caller.src = CC.ir_to_codeinf!(src)
        frame.edges = collect(Any, caller.src.edges)
    end

    return @invoke CC.finish!(analyzer::AbstractAnalyzer, frame::CC.InferenceState, validation_world::UInt, time_before::UInt64)
end

function gpu_report_runtime_dispatch!(analyzer::GPUAnalyzer, caller::CC.InferenceResult, state::JET.OptimizedIRState)
    for (pc, inst) in enumerate(state.ir.stmts)
        x = inst[:stmt]
        if Base.Meta.isexpr(x, :call)
            ft = CC.widenconst(CC.argextype(first(x.args), state.ir))
            ft <: Core.Builtin && continue
            add_new_report!(analyzer, caller, GPURuntimeDispatchReport((state, pc)))
        end
    end
end

function gpu_report_throw_statements!(analyzer::GPUAnalyzer, caller::CC.InferenceResult, state::JET.OptimizedIRState)
    mi = caller.linfo
    def = mi.def
    caller_module = def isa Method ? def.module : def

    for (pc, inst) in enumerate(state.ir.stmts)
        x = inst[:stmt]
        is_throw = Base.Meta.isexpr(x, :throw)
        if !is_throw && (Base.Meta.isexpr(x, :invoke) || Base.Meta.isexpr(x, :call))
            args = x.args
            if length(args) >= 1
                ft = Base.Meta.isexpr(x, :invoke) && length(args) >= 2 ?
                    CC.argextype(args[2], state.ir) :
                    CC.argextype(first(args), state.ir)
                f = CC.singleton_type(ft)
                if f in (Base.throw, Core.throw, Base.error, Base.ArgumentError,
                         Base.BoundsError, Base.DomainError)
                    is_throw = true
                end
            end
        end
        if is_throw
            analyzer.ignore_intrinsic_throws && caller_module in GPU_INTRINSIC_MODULES && continue
            add_new_report!(analyzer, caller, GPUThrowStatementReport((state, pc)))
        end
    end
end

function gpu_report_allocations!(analyzer::GPUAnalyzer, caller::CC.InferenceResult, state::JET.OptimizedIRState)
    for (pc, inst) in enumerate(state.ir.stmts)
        x = inst[:stmt]
        alloc_type = nothing
        if Base.Meta.isexpr(x, :new) && length(x.args) >= 1
            T = CC.widenconst(CC.argextype(x.args[1], state.ir))
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
        alloc_type !== nothing && add_new_report!(analyzer, caller, GPUAllocationReport((state, pc), alloc_type))
    end
end

const _gpu_analysis_token = AnalysisToken()

function GPUAnalyzer(world::UInt = Base.get_world_counter();
    analysis_token::AnalysisToken = _gpu_analysis_token,
    method_table::CC.MethodTableView = CC.InternalMethodTable(world),
    frame_filter = (_::Core.MethodInstance) -> true,
    check_dispatch::Bool = true,
    check_throw::Bool = true,
    check_alloc::Bool = true,
    ignore_intrinsic_throws::Bool = true,
    jetconfigs...)
    state = AnalyzerState(world; jetconfigs...)
    GPUAnalyzer(state, analysis_token, method_table, frame_filter, check_dispatch, check_throw, check_alloc,
                ignore_intrinsic_throws, BitVector())
end

function report_gpu_compat(args...; jetconfigs...)
    analyzer = GPUAnalyzer(; jetconfigs...)
    JET.analyze_and_report_call!(analyzer, args...; jetconfigs...)
end

gpu_module_filter(m::Module) = (mi::Core.MethodInstance) -> begin
    def = mi.def
    (def isa Method ? def.module : def) === m
end

"""The interpreter `f(types...)` is compiled through on `device`."""
device_interpreter(device, f, types) = Mantle.kernelinterpreter(device, f, Tuple{types...})

"""
Test that a function has no runtime dispatch (GPU compatible), inferred as `device` compiles it.
Returns true if no dispatch issues found.
"""
function test_no_dispatch(f, types; device = Hikari.mantle_device(Mantle.defaultbackend()),
                          frame_filter = gpu_module_filter(Hikari))
    method_table = CC.method_table(device_interpreter(device, f, types))
    result = report_gpu_compat(f, types; method_table, frame_filter, check_throw=false, check_alloc=false)
    reports = JET.get_reports(result)
    dispatch_count = count(r -> r isa GPURuntimeDispatchReport, reports)
    dispatch_count == 0
end

# ============================================================================
# Test Scene Generator
# ============================================================================

using Adapt

function gen_gpu_test_scene()
    # Materials
    white_matte = Hikari.Diffuse(color = Hikari.RGBSpectrum(0.73f0))
    glass = Hikari.Glass(ior = 1.5f0)
    emissive = Hikari.Emission(color = Hikari.RGBSpectrum(10f0))

    # Media
    fog = Hikari.HomogeneousMedium(
        σ_a=Hikari.RGBSpectrum(0.01f0),
        σ_s=Hikari.RGBSpectrum(0.1f0),
        Le = Hikari.RGBSpectrum(0f0),
        g=0.3f0
    )
    glass_with_fog = Hikari.MediumInterface(glass; inside=fog, outside=nothing)

    # Build scene using push! API, on the device the kernels run on
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend)
    push!(scene, normal_mesh(Rect3f(Vec3f(-1, 0, -1), Vec3f(2, 0.01f0, 2))), white_matte)
    push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0, 0.5f0, 0), 0.3f0), 16)), glass_with_fog)
    push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0, 1.5f0, 0), 0.1f0), 16)), emissive)

    # Lights
    push!(scene, Hikari.PointLight(Point3f(0f0, 2f0, 0f0), Hikari.RGBSpectrum(10f0)))
    push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.1f0)))
    Hikari.sync!(scene)

    # Adapt scene to get StaticMultiTypeSet for type analysis
    adapted = Adapt.adapt(backend, scene)

    # Get rgb2spec_table directly
    memory = Hikari.DeviceMemory(backend)
    rgb2spec_table = Hikari.to_gpu(memory, Hikari.get_srgb_table())

    (
        scene=adapted,
        materials=adapted.materials,
        media=adapted.media,
        lights=adapted.lights,
        rgb2spec_table=rgb2spec_table,
        memory=memory,
        lambda=Hikari.SampledWavelengths{4}((400f0, 500f0, 600f0, 700f0), (1f0, 1f0, 1f0, 1f0)),
        mat_idx=Hikari.SetKey(UInt32(1), UInt32(1)),
        ray=Raycore.Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=Inf32),
    )
end

# ============================================================================
# Tests
# ============================================================================

dispatching_probe(x::Vector{Any}) = sin(x[1]) + 1f0

@testset "GPU Compatibility: VolPath Kernel Functions" begin
    td = gen_gpu_test_scene()

    tfc_type = Hikari.TextureFilterContext

    # NOTE: dropped the "Material Dispatch Functions" testset — the
    # `*_dispatch` helpers it covered were removed in the
    # physical-wavefront → volpath refactor (commit ee442ef). The volpath
    # integrator now reaches these predicates inline via `with_index`, and
    # the per-material `is_emissive(::Mat)` / `is_pure_emissive(::Mat)`
    # methods are already exercised by the higher-level dispatch tests
    # below (BSDF Sampling and Microfacet/Fresnel).

    # The check can fail: a call on an element of a `Vector{Any}` is a runtime dispatch.
    @testset "a dispatch is reported" begin
        @test !test_no_dispatch(dispatching_probe, (Vector{Any},); frame_filter = gpu_module_filter(@__MODULE__))
    end

    @testset "Phase Functions" begin
        @test test_no_dispatch(Hikari.hg_p, (Float32, Float32))
        @test test_no_dispatch(Hikari.sample_hg, (Float32, Vec3f, Point2f))
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
        # What the shading kernels sample: the BSDF each material of the scene builds at a hit
        # (`material_bxdf`), reading the view of the materials its kernel takes (`shade_set`).
        # It asked `get_bxdf`, which a shader material does not build its BSDF with: the glass,
        # a Principled, came back as itself, and no BSDF method took it.
        # The types are the kernel's: device arrays, which index only under the device's
        # method table. With the host's arrays and table, the glass's inference ended at its
        # program's first read, and a check found nothing to report.
        device = Hikari.mantle_device(Mantle.defaultbackend())
        table, lambda = Mantle.devicetype(device, td.rgb2spec_table), typeof(td.lambda)
        @testset "$(nameof(eltype(v)))" for v in td.materials.data
            textures = Mantle.devicetype(device, Hikari.shade_set(td.materials, eltype(v)))
            build = (eltype(v), table, textures, Hikari.VPMaterialEvalWorkItem, Hikari.PerspectiveCamera, Int32, lambda, Bool)
            @test test_no_dispatch(Hikari.material_bxdf, build)
            bsdf = only(Base.return_types(Hikari.material_bxdf, build;
                                          interp = device_interpreter(device, Hikari.material_bxdf, build)))
            @test isconcretetype(bsdf)
            @test test_no_dispatch(Hikari.sample_bsdf_spectral,
                (bsdf, table, textures, Vec3f, Vec3f, Vec3f, tfc_type, lambda, Point2f, Float32, Bool))
            @test test_no_dispatch(Hikari.evaluate_bsdf_spectral,
                (bsdf, table, textures, Vec3f, Vec3f, Vec3f, Vec3f, tfc_type, lambda, Bool))
        end
    end

    @testset "Microfacet/Fresnel Functions" begin
        @test test_no_dispatch(Hikari.trowbridge_reitz_sample_wm, (Vec3f, Point2f, Float32, Float32))
    end
end
