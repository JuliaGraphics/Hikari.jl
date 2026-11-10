module SretContainerProbe
using Lava, LLVM, Test

const IR = """
@out = internal addrspace(3) global [64 x float] undef
define internal void @pair(ptr noalias nocapture sret([2 x float]) %dst, float %x) noinline {
entry:
  %twice = fmul float %x, 2.0
  %a = insertvalue [2 x float] undef, float %x, 0
  %b = insertvalue [2 x float] %a, float %twice, 1
  store [2 x float] %b, ptr %dst, align 4
  ret void
}
define float @caller(float %x) noinline {
entry:
  %box = alloca [1 x i64], align 8
  call void @pair(ptr %box, float %x)
  %second = getelementptr i8, ptr %box, i64 4
  %value = load float, ptr %second, align 4
  ret float %value
}
define void @main() {
entry:
  %value = call float @caller(float 1.0)
  %dst = getelementptr [64 x float], ptr addrspace(3) @out, i64 0, i64 0
  store float %value, ptr addrspace(3) %dst
  ret void
}
"""

function run()
    @testset "normalize owned result layout before deciding to inline" begin
        for normalize in (false, true)
            LLVM.Context() do ctx
                mod = parse(LLVM.Module, IR)
                normalize && Lava.type_sret_boxes!(mod, mod.datalayout)
                mismatch = Lava.has_alloca_type_mismatch_at_callsites(mod.functions["pair"], mod)
                @test mismatch == !normalize
                Lava.force_inline_all!(mod, mod.functions["main"])
                retained = haskey(mod.functions, "pair") && !isempty(mod.functions["pair"].blocks)
                @test retained == normalize
                Lava.run_llvm_passes!(mod, mod.functions["main"])
                bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
                @test Lava.validate_spirv(bytes) === nothing
                name = gensym(:sret_cpu)
                source = string(mod)
                cpu = Core.eval(@__MODULE__, :(function $name(x::Float32)
                    Base.llvmcall(($source, "caller"), Float32, Tuple{Float32}, x)
                end))
                @test [Base.invokelatest(cpu, x) for x in (-2f0, 0f0, 0.5f0, 7f0)] == Float32[-4, 0, 1, 14]
                println((; normalize, retained, spirv_bytes = length(bytes)))
            end
        end
    end
    return nothing
end
end
