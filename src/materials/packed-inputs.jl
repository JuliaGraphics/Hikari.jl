# Resolve constant Principled inputs once on the host. Computed inputs keep the
# same gathered-stack indices, so graphs still share their shading kernel.
const PACKED_PRINCIPLED_INPUTS = 0xfd
# Field order of PrincipledInputs in the original 34-argument node. Emission
# strength (31) is stored separately; argument 12 is unused by that structure.
const PRINCIPLED_INPUT_ARGS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 13, 34, 10, 11)
const PRINCIPLED_PACKED_PAYLOAD = fieldoffset(NodeInstr{NodeFeature.CLOSURES}, 7)
const PRINCIPLED_PACKED_CODES = PRINCIPLED_PACKED_PAYLOAD + sizeof(PrincipledInputs)
const PRINCIPLED_PACKED_WIDTHS = PRINCIPLED_PACKED_CODES + 34 - NODE_ARGS
const PRINCIPLED_PACKED_STRENGTH = sizeof(NodeInstr{NodeFeature.CLOSURES}) * instruction_count(PrincipledBSDF) - sizeof(Float32)
struct ZeroInputStack end
@inline _get(::ZeroInputStack, offset) = 0.0f0

function pack_principled_inputs(instructions)
    result = copy(instructions)
    bytes = reinterpret(UInt8, result)
    n = Vec3f(0, 0, 1)
    sc = ShaderContext(TextureFilterContext(Point2f(0), Point3f(0), n), n, Vec3f(0), Vec3f(0),
        Vec3f(1, 0, 0), n, 1f0, Vec3f(0), Point2f(0))
    empty_stack = ZeroInputStack()
    i = Int32(1)
    while instructions[i].op != NodeOp.RESULT
        ins = instructions[i]
        chunks = surface_chunks(ins)
        after = next_node(instructions, i, chunks)
        if ins.op == NodeOp.PRINCIPLED
            codes = UInt8[]
            widths = UInt64(0)
            for j in 1:34
                arg_ins = instructions[i + (j - 1) ÷ NODE_ARGS]
                a = (j - 1) % NODE_ARGS + 1
                code = arg_ins.arg[a]
                push!(codes, reads_stack(code) || code in (SHADING_NORMAL_ARG, SHADING_TANGENT_ARG) ? code : NO_SLOT)
                _bit(arg_ins.wide, a) && (widths |= UInt64(1) << (j - 1))
            end
            p = principled_inputs(empty_stack, instructions, i, sc)
            # Store emission before multiplying its strength: either input may
            # be computed independently, including a strength that is zero.
            values = ntuple(Val(fieldcount(PrincipledInputs))) do j
                fieldname(PrincipledInputs, j) === :emission ? node_v(empty_stack, instructions, i, 30, chunks, sc) : getfield(p, j)
            end
            p = PrincipledInputs(values...)
            strength = node_f(empty_stack, instructions, i, 31, chunks, sc)
            flags = UInt8((codes[7] == SHADING_NORMAL_ARG) |
                ((codes[26] == SHADING_NORMAL_ARG) << 1) | ((codes[18] == SHADING_TANGENT_ARG) << 2))
            dynamic = any(enumerate(codes)) do (j, code)
                code != NO_SLOT && !((j in (7, 26) && code == SHADING_NORMAL_ARG) ||
                    (j == 18 && code == SHADING_TANGENT_ARG))
            end
            out = (PACKED_PRINCIPLED_INPUTS, flags, UInt8(dynamic), ins.out[4], ins.out[5])
            result[i] = typeof(ins)(ins.op, ins.mode, out, ins.outwide, Tuple(codes[1:NODE_ARGS]), UInt16(widths & 0xffff), ins.k)
            start = (Int(i) - 1) * sizeof(eltype(result)) + 1
            PRINCIPLED_PACKED_WIDTHS + 3 <= PRINCIPLED_PACKED_STRENGTH || error("Principled input record exceeds its instruction payload")
            copyto!(bytes, start + PRINCIPLED_PACKED_PAYLOAD, reinterpret(UInt8, [p]), 1, sizeof(p))
            copyto!(bytes, start + PRINCIPLED_PACKED_CODES, codes, NODE_ARGS + 1, 34 - NODE_ARGS)
            for k in 0:2
                bytes[start + PRINCIPLED_PACKED_WIDTHS + k] = UInt8((widths >> (16 + 8k)) & 0xff)
            end
            copyto!(bytes, start + PRINCIPLED_PACKED_STRENGTH, reinterpret(UInt8, [strength]), 1, sizeof(strength))
        end
        i = after
    end
    return result
end

# The packed record is read in bytes of instruction `i`, through the pointer kind the
# program array has: a `Ptr` on Lava, an address-space `LLVMPtr` on Metal, which has
# no conversion to `Ptr` (a `MethodError` left in every Principled shading kernel).
@inline recordbytes(prog, i) = bytepointer(pointer(prog, i))
@inline bytepointer(p::Ptr) = Ptr{UInt8}(p)
@inline bytepointer(p::Core.LLVMPtr{T, A}) where {T, A} = reinterpret(Core.LLVMPtr{UInt8, A}, p)
@inline loadat(::Type{T}, p::Ptr{UInt8}, offset) where {T} = unsafe_load(Ptr{T}(p + offset))
@inline loadat(::Type{T}, p::Core.LLVMPtr{UInt8, A}, offset) where {T, A} =
    unsafe_load(reinterpret(Core.LLVMPtr{T, A}, p + offset))

@inline function packed_principled_code(prog, i, ::Val{J}) where {J}
    if J <= NODE_ARGS
        return @inbounds prog[i].arg[J]
    end
    return loadat(UInt8, recordbytes(prog, i), PRINCIPLED_PACKED_CODES + J - NODE_ARGS - 1)
end

@inline function packed_principled_value(::Type{Float32}, constant, s, prog, i, sc, j::Val, widths)
    code = packed_principled_code(prog, i, j)
    code == NO_SLOT && return constant
    code == SHADING_NORMAL_ARG && return sc.tfc.n[1]
    code == SHADING_TANGENT_ARG && return sc.dpdus[1]
    return _get(s, code)
end
@inline function packed_principled_value(::Type{Vec3f}, constant, s, prog, i, sc, j::Val{J}, widths) where {J}
    code = packed_principled_code(prog, i, j)
    code == NO_SLOT && return constant
    code == SHADING_NORMAL_ARG && return sc.tfc.n
    code == SHADING_TANGENT_ARG && return sc.dpdus
    iszero(widths & (UInt64(1) << (J - 1))) && return _splat(_get(s, code))
    return Vec3f(_get(s, code), _get(s, code + 0x01), _get(s, code + 0x02))
end

@generated function packed_principled_dynamic(p::PrincipledInputs, s, prog, i, sc, widths)
    fields = [:(packed_principled_value($(fieldtype(PrincipledInputs, j)), getfield(p, $j),
        s, prog, i, sc, Val($(PRINCIPLED_INPUT_ARGS[j])), widths)) for j in 1:fieldcount(PrincipledInputs)]
    return quote
        $(Expr(:meta, :inline))
        PrincipledInputs($(fields...))
    end
end

@generated function packed_principled_finish(p::PrincipledInputs, sc, flags::UInt8, strength::Float32)
    fields = map(fieldnames(PrincipledInputs)) do name
        name === :n && return :((flags & 0x01) != 0 ? sc.tfc.n : p.n)
        name === :coat_n && return :((flags & 0x02) != 0 ? sc.tfc.n : p.coat_n)
        name === :tangent && return :((flags & 0x04) != 0 ? sc.dpdus : p.tangent)
        name === :emission && return :(p.emission * strength)
        return :(getfield(p, $(QuoteNode(name))))
    end
    return quote
        $(Expr(:meta, :inline))
        PrincipledInputs($(fields...))
    end
end

@inline function packed_principled_inputs(s, prog, i::Int32, sc::ShaderContext)
    @inbounds ins = prog[i]
    bytes = recordbytes(prog, i)
    p = loadat(PrincipledInputs, bytes, PRINCIPLED_PACKED_PAYLOAD)
    strength = loadat(Float32, bytes, PRINCIPLED_PACKED_STRENGTH)
    if ins.out[3] != 0x00
        widths = UInt64(ins.wide)
        for k in 0:2
            widths |= UInt64(loadat(UInt8, bytes, PRINCIPLED_PACKED_WIDTHS + k)) << (16 + 8k)
        end
        p = packed_principled_dynamic(p, s, prog, i, sc, widths)
        strength = packed_principled_value(Float32, strength, s, prog, i, sc, Val(31), widths)
    end
    return packed_principled_finish(p, sc, ins.out[2], strength)
end
