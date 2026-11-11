module AllocationRetypeProbe
using Hikari, Mantle, Revise, JSON3

# Diagnostic only: bisect allocation rewrites without changing material math.
# The old method world runs the original implementation for retained cases.
function run(skip, output; definition = nothing)
    L = Mantle.Lava
    original = Revise.CodeTracking.definition(Expr, first(methods(L.rewrite_alloca!)))
    original isa Expr || error("Cannot recover allocation rewrite")
    previous = L._initialization_world[]
    original_world = Base.get_world_counter()
    try
        replacement = definition === nothing ? quote
            function rewrite_alloca!(info::AllocaInfo, T::LLVM.LLVMType, dl::LLVM.DataLayout)
                Base.invokelatest($skip, info) && return false
                Base.invoke_in_world($original_world, rewrite_alloca!, info, T, dl)
            end
        end : definition
        Core.eval(L, replacement)
        L._initialization_world[] = Base.get_world_counter()
        withenv("LAVA_DUMP_KERNELS" => "1", "LAVA_DUMP_KERNELS_DIR" => output * "_modules") do
            method = first(filter(m -> occursin("Looped", string(m.sig)), methods(Hikari.surface_setup_slots)))
            Core.eval(Hikari, Revise.CodeTracking.definition(Expr, method))
            cases = [p for batch in Main.cycles_batches() for p in batch
                     if p[1] in ("subsurface_node_zero_channel", "mix_subsurface_diffuse")]
            image = Base.invokelatest(Main.render_cycles_batch, cases; samples = 64)
            means = [Main.cycles_mean(image, Main.cycles_sphere_pixels(k)) for k in eachindex(cases)]
            open(io -> JSON3.write(io, (; cases = first.(cases), means)), output * ".json", "w")
            println("RETYPE_CONTROL ", output, " ", means); flush(stdout)
        end
    finally
        Core.eval(L, original)
        L._initialization_world[] = previous
    end
    nothing
end

function grouped_definition(x)
    x isa Expr || return x
    if x.head == :(=) && x.args[1] == :new_arr_ty
        return quote
            old_ty = LLVM.LLVMType(API.LLVMGetAllocatedType(info.alloca))
            group = old_ty isa LLVM.ArrayType && old_ty.element_type isa LLVM.IntegerType &&
                old_ty.element_type.width == 64 && sz == 4 ? 2 : 1
            new_arr_ty = group == 1 ? LLVM.ArrayType(T, n_elems) : LLVM.ArrayType(LLVM.VectorType(T, group), n_elems ÷ group)
        end
    elseif x.head == :return && occursin("LLVM.gep!(builder, new_arr_ty", string(x))
        return quote
            if group == 1
                return LLVM.gep!(builder, new_arr_ty, new_alloca, LLVM.Value[zero32, idx_value])
            end
            word = LLVM.ashr!(builder, idx_value, LLVM.ConstantInt(i64, 1))
            lane = LLVM.and!(builder, idx_value, LLVM.ConstantInt(i64, 1))
            return LLVM.gep!(builder, new_arr_ty, new_alloca, LLVM.Value[zero32, word, lane])
        end
    end
    return Expr(x.head, (grouped_definition(a) for a in x.args)...)
end
end
