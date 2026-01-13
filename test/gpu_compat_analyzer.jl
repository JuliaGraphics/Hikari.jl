# GPU Compatibility Analyzer for KernelAbstractions.jl kernels
# Detects patterns that are incompatible with GPU execution:
# - Dynamic dispatch (runtime method lookup)
# - Throw statements / exception handling
# - Heap allocations
#
# Based on JET.jl's pluggable analyzer framework
#
# Usage:
#   using JET
#   include("gpu_compat_analyzer.jl")
#   @report_gpu_compat my_kernel_function(args...)
#
# Or for testing:
#   @test_gpu_compat my_kernel_function(args...)

using JET.JETInterface
using JET: JET, CC
using InteractiveUtils
using Test

# ============================================================================
# Analyzer Type
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

# AbstractAnalyzer API requirements
JETInterface.AnalyzerState(analyzer::GPUAnalyzer) = analyzer.state
JETInterface.AbstractAnalyzer(analyzer::GPUAnalyzer, state::AnalyzerState) =
    GPUAnalyzer(state, analyzer.analysis_token, analyzer.frame_filter,
                analyzer.check_dispatch, analyzer.check_throw, analyzer.check_alloc,
                analyzer.ignore_intrinsic_throws, BitVector())
JETInterface.AnalysisToken(analyzer::GPUAnalyzer) = analyzer.analysis_token

# ============================================================================
# Report Types
# ============================================================================

@jetreport struct GPURuntimeDispatchReport <: InferenceErrorReport end
function JETInterface.print_report_message(io::IO, ::GPURuntimeDispatchReport)
    print(io, "runtime dispatch detected (GPU incompatible)")
end

@jetreport struct GPUThrowStatementReport <: InferenceErrorReport end
function JETInterface.print_report_message(io::IO, ::GPUThrowStatementReport)
    print(io, "throw statement detected (GPU incompatible)")
end

@jetreport struct GPUAllocationReport <: InferenceErrorReport
    alloc_type::String
end
function JETInterface.print_report_message(io::IO, r::GPUAllocationReport)
    print(io, "heap allocation detected: ", r.alloc_type, " (GPU incompatible)")
end

@jetreport struct GPUOptFailureReport <: InferenceErrorReport end
function JETInterface.print_report_message(io::IO, ::GPUOptFailureReport)
    print(io, "optimization failed (GPU incompatible)")
end

# ============================================================================
# GPU-Safe Intrinsic Modules
# ============================================================================

# These modules contain functions that GPUCompiler replaces with device intrinsics.
# Throws from these modules are dead code on GPU and should be ignored.
const GPU_INTRINSIC_MODULES = Set([
    Base.Math,  # sqrt, sin, cos, etc domain errors
])

# ============================================================================
# Compiler Hooks
# ============================================================================

function CC.finish!(analyzer::GPUAnalyzer, frame::CC.InferenceState, validation_world::UInt, time_before::UInt64)
    caller = frame.result
    src = caller.src
    if src isa CC.OptimizationState
        caller.src = CC.ir_to_codeinf!(src)
        frame.edges = collect(Any, caller.src.edges)
    end

    if analyzer.frame_filter(frame.linfo)
        if isa(src, Core.Const)
            # Optimization was very successful, nothing to report
        elseif isnothing(src)
            # Compiler decides not to do optimization
            add_new_report!(analyzer, caller, GPUOptFailureReport(caller.linfo))
        elseif isa(src, CC.OptimizationState)
            # Compiler optimized it, analyze it
            analyzer.check_dispatch && gpu_report_runtime_dispatch!(analyzer, caller, src)
            analyzer.check_throw && gpu_report_throw_statements!(analyzer, caller, src)
            analyzer.check_alloc && gpu_report_allocations!(analyzer, caller, src)
        end
    end

    return @invoke CC.finish!(analyzer::AbstractAnalyzer, frame::CC.InferenceState, validation_world::UInt, time_before::UInt64)
end

# ============================================================================
# Analysis Functions
# ============================================================================

"""
    gpu_report_runtime_dispatch!(analyzer, caller, opt)

Detect `:call` expressions in optimized IR (indicates runtime dispatch).
In optimized code, `:call` means runtime method lookup, while `:invoke` means
the method was statically resolved.
"""
function gpu_report_runtime_dispatch!(analyzer::GPUAnalyzer, caller::CC.InferenceResult, opt::CC.OptimizationState)
    (; sptypes, slottypes) = opt
    for (pc, x) in enumerate(opt.src.code)
        if Base.Meta.isexpr(x, :call)
            ft = CC.widenconst(CC.argextype(first(x.args), opt.src, sptypes, slottypes))
            ft <: Core.Builtin && continue  # Ignore calls to builtin intrinsics
            add_new_report!(analyzer, caller, GPURuntimeDispatchReport((opt, pc)))
        end
    end
end

"""
    gpu_report_throw_statements!(analyzer, caller, opt)

Detect throw statements and error calls in optimized IR.

By default, ignores throws from GPU intrinsic modules (like Base.Math) because
GPUCompiler replaces these functions with device intrinsics, making the throw
branches dead code.
"""
function gpu_report_throw_statements!(analyzer::GPUAnalyzer, caller::CC.InferenceResult, opt::CC.OptimizationState)
    (; sptypes, slottypes) = opt

    # Get caller module to check if throws are from GPU-safe intrinsics
    mi = caller.linfo
    def = mi.def
    caller_module = def isa Method ? def.module : def

    for (pc, x) in enumerate(opt.src.code)
        is_throw = false

        if Base.Meta.isexpr(x, :throw)
            is_throw = true
        elseif Base.Meta.isexpr(x, :invoke) || Base.Meta.isexpr(x, :call)
            args = x.args
            if length(args) >= 1
                ft = if Base.Meta.isexpr(x, :invoke)
                    length(args) >= 2 ? CC.argextype(args[2], opt.src, sptypes, slottypes) : nothing
                else
                    CC.argextype(first(args), opt.src, sptypes, slottypes)
                end

                if ft !== nothing
                    f = CC.singleton_type(ft)
                    if f !== nothing
                        if f === Base.throw || f === Core.throw ||
                           f === Base.error || f === Base.ArgumentError ||
                           f === Base.BoundsError || f === Base.DomainError
                            is_throw = true
                        end
                    end
                end
            end
        end

        if is_throw
            # Filter GPU-safe throws from intrinsic modules (sqrt, sin, cos, etc.)
            if analyzer.ignore_intrinsic_throws && caller_module in GPU_INTRINSIC_MODULES
                continue
            end
            add_new_report!(analyzer, caller, GPUThrowStatementReport((opt, pc)))
        end
    end
end

"""
    gpu_report_allocations!(analyzer, caller, opt)

Detect heap allocations in optimized IR.
Checks for:
- :new expressions for mutable structs
- Core.memorynew calls (array backing memory in Julia 1.11+)
- Foreign calls to jl_alloc* functions
"""
function gpu_report_allocations!(analyzer::GPUAnalyzer, caller::CC.InferenceResult, opt::CC.OptimizationState)
    (; sptypes, slottypes) = opt
    for (pc, x) in enumerate(opt.src.code)
        alloc_type = nothing

        # Check for :new expressions (mutable struct allocation, including Vector)
        if Base.Meta.isexpr(x, :new)
            if length(x.args) >= 1
                T = CC.argextype(x.args[1], opt.src, sptypes, slottypes)
                T_concrete = CC.widenconst(T)
                if T_concrete isa DataType && Base.ismutabletype(T_concrete)
                    alloc_type = "mutable struct $(T_concrete.name.name)"
                end
            end
        # Check for :call expressions that allocate memory
        elseif Base.Meta.isexpr(x, :call) && length(x.args) >= 1
            first_arg = x.args[1]
            # Check for GlobalRef to Core.memorynew (Julia 1.11+)
            if first_arg isa GlobalRef
                if first_arg.mod === Core && first_arg.name === :memorynew
                    alloc_type = "memory allocation (Core.memorynew)"
                end
            end
        # Check for array allocation foreign calls (older Julia versions)
        elseif Base.Meta.isexpr(x, :foreigncall)
            fname = x.args[1]
            if fname isa QuoteNode
                fname = fname.value
            end
            if fname isa Symbol
                fname_str = String(fname)
                if startswith(fname_str, "jl_alloc") ||
                   fname_str == "jl_array_copy" ||
                   fname_str == "jl_new_array"
                    alloc_type = "array ($fname_str)"
                end
            end
        end

        if alloc_type !== nothing
            add_new_report!(analyzer, caller, GPUAllocationReport((opt, pc), alloc_type))
        end
    end
end

# ============================================================================
# Entry Points
# ============================================================================

const global_gpu_analysis_token = AnalysisToken()

"""
    GPUAnalyzer(; kwargs...)

Create a GPU compatibility analyzer.

# Keyword Arguments
- `frame_filter`: Predicate `(MethodInstance) -> Bool` to filter which frames to analyze
- `check_dispatch::Bool = true`: Check for runtime dispatch
- `check_throw::Bool = true`: Check for throw statements
- `check_alloc::Bool = true`: Check for heap allocations
- `ignore_intrinsic_throws::Bool = true`: Ignore throws from GPU intrinsic modules (Base.Math)
- `jetconfigs...`: Additional JET configuration options
"""
function GPUAnalyzer(world::UInt = Base.get_world_counter();
    analysis_token::AnalysisToken = global_gpu_analysis_token,
    frame_filter = x::Core.MethodInstance -> true,
    check_dispatch::Bool = true,
    check_throw::Bool = true,
    check_alloc::Bool = true,
    ignore_intrinsic_throws::Bool = true,
    jetconfigs...)
    state = AnalyzerState(world; jetconfigs...)
    return GPUAnalyzer(state, analysis_token, frame_filter, check_dispatch, check_throw, check_alloc, ignore_intrinsic_throws, BitVector())
end

"""
    report_gpu_compat(f, [types]; jetconfigs...) -> JETCallResult

Analyze a function call for GPU compatibility issues.

# Example
```julia
report_gpu_compat(my_kernel_inner!, (Vector{Float32}, Vector{Float32}, Int))
report_gpu_compat(my_func, (Int,); check_throw=false)  # Ignore throws
report_gpu_compat(sqrt, (Float32,); ignore_intrinsic_throws=false)  # Show all throws
```
"""
function report_gpu_compat(args...; jetconfigs...)
    @nospecialize args jetconfigs
    analyzer = GPUAnalyzer(; jetconfigs...)
    return JET.analyze_and_report_call!(analyzer, args...; jetconfigs...)
end

"""
    @report_gpu_compat [jetconfigs...] f(args...)

Macro version of `report_gpu_compat`. Evaluates arguments and analyzes the call.

# Example
```julia
@report_gpu_compat my_kernel_inner!(output, input, 1)
@report_gpu_compat frame_filter=gpu_module_filter(MyModule) my_func(x)
```
"""
macro report_gpu_compat(ex0...)
    return InteractiveUtils.gen_call_with_extracted_types_and_kwargs(__module__, :report_gpu_compat, ex0)
end

"""
    @test_gpu_compat [jetconfigs...] [broken=false] [skip=false] f(args...)

Test that a function call is GPU-compatible (no runtime dispatch, throws, or allocations).

Integrates with Julia's Test stdlib.

# Example
```julia
@testset "GPU Compatibility" begin
    @test_gpu_compat my_kernel_inner!(output, input, 1)
    @test_gpu_compat frame_filter=gpu_module_filter(Hikari) other_func(x)
end
```
"""
macro test_gpu_compat(ex0...)
    return JET.call_test_ex(:report_gpu_compat, Symbol("@test_gpu_compat"), ex0, __module__, __source__)
end

"""
    test_gpu_compat(f, [types]; broken=false, skip=false, jetconfigs...)

Function version of `@test_gpu_compat`.
"""
function test_gpu_compat(@nospecialize(args...); jetconfigs...)
    return JET.func_test(report_gpu_compat, :test_gpu_compat, args...; jetconfigs...)
end

# ============================================================================
# Helper: Module Filter
# ============================================================================

"""
    gpu_module_filter(m::Module)

Create a frame filter that only analyzes code from module `m`.

# Example
```julia
# Only report issues in Hikari module, not in Base or other dependencies
@report_gpu_compat frame_filter=gpu_module_filter(Hikari) my_func(x)
```
"""
function gpu_module_filter(m::Module)
    return function(mi::Core.MethodInstance)
        def = mi.def
        isa(def, Method) ? def.module === m : def === m
    end
end
