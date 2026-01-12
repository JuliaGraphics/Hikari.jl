# TextureRef - GPU-compatible texture reference
# Replaces Texture{T, N, Array} on GPU to avoid SPIR-V pointer-in-composite issues

"""
    TextureRef{T}

GPU-side texture reference. Stores either:
- A constant value (texture_type == 0)
- An index into a texture tuple (texture_type > 0)

This avoids storing CLDeviceArray directly in material structs, which causes
SPIR-V validation errors when the material is stored in an intermediate variable.
"""
struct TextureRef{T}
    texture_type::UInt8    # Which tuple slot (1-based), 0 = constant
    texture_idx::UInt32    # Index within that texture type's array
    const_value::T         # Constant value (used when texture_type == 0)
end

# Constant texture constructor
TextureRef(value::T) where T = TextureRef{T}(UInt8(0), UInt32(0), value)

# Indexed texture constructor
TextureRef{T}(type::UInt8, idx::UInt32) where T = TextureRef{T}(type, idx, zero(T))

# Check if constant
@inline is_constant_texture(ref::TextureRef) = ref.texture_type == UInt8(0)

# ============================================================================
# Texture Dispatch
# ============================================================================

"""
    evaluate_texture_ref(textures::NTuple{N}, ref::TextureRef{T}, uv::Point2f) -> T

Evaluate a TextureRef - either return constant or dispatch to texture tuple.
Uses @generated for type-stable dispatch over the texture tuple.
"""
@propagate_inbounds @generated function evaluate_texture_ref(
    textures::NTuple{N,Any}, ref::TextureRef{T}, uv::Point2f
) where {N, T}
    # Each textures[$i] is an array of matrices (texture data), not Texture structs.
    # This avoids loading structs with nested device array pointers on GPU.
    branches = [quote
        if ref.texture_type === UInt8($i)
            @inbounds data = textures[$i][ref.texture_idx]
            # Sample the texture data directly (UV already adjusted by caller if needed)
            return _sample_texture_data_direct(data, uv)::$T
        end
    end for i in 1:N]

    quote
        # Fast path: constant texture
        if ref.texture_type === UInt8(0)
            return ref.const_value
        end
        $(branches...)
        return ref.const_value  # Fallback (should never reach)
    end
end

# Fallback for empty texture tuple (all textures are constant)
@propagate_inbounds function evaluate_texture_ref(
    ::Tuple{}, ref::TextureRef{T}, ::Point2f
) where T
    return ref.const_value
end

# ============================================================================
# Unified Texture Evaluation - Works for both Texture and TextureRef
# ============================================================================

"""
    eval_tex(textures, tex_or_ref, uv::Point2f) -> T

Unified texture evaluation that works for both:
- CPU: `tex_or_ref` is a `Texture{T}`, `textures` is ignored
- GPU: `tex_or_ref` is a `TextureRef{T}`, dispatches through `textures` tuple

This allows the same material evaluation code to work on both CPU and GPU.
"""
# CPU path: Texture (ignores textures tuple)
@propagate_inbounds function eval_tex(::Any, tex::Texture{T}, uv::Point2f) where T
    return evaluate_texture(tex, uv)
end

# GPU path: TextureRef (uses textures tuple)
@propagate_inbounds function eval_tex(textures::Tuple, ref::TextureRef{T}, uv::Point2f) where T
    return evaluate_texture_ref(textures, ref, uv)
end

# ============================================================================
# Texture Collector - For building texture tuple during to_gpu
# ============================================================================

"""
    TextureCollector

Collects texture DATA (matrices) during material conversion and builds a typed tuple.
Used during to_gpu to extract texture data from materials.

IMPORTANT: We store raw texture data matrices, NOT Texture structs.
This allows GPU storage as device arrays without nested pointer issues.
Textures are grouped by their data's concrete matrix type (element type + array type).
"""
mutable struct TextureCollector
    # Map from data matrix type to slot_index
    type_to_slot::Dict{DataType, UInt8}
    type_order::Vector{DataType}
    # Stores the actual data matrices (not Texture structs)
    data_by_type::Dict{DataType, Vector{Any}}
end

TextureCollector() = TextureCollector(
    Dict{DataType, UInt8}(),
    DataType[],
    Dict{DataType, Vector{Any}}()
)

"""
    register_texture!(collector::TextureCollector, tex::Texture{T}) -> (UInt8, UInt32)

Register a texture's DATA and return its (type_slot, index) for building TextureRef.
Only the data matrix is stored, not the Texture struct wrapper.
"""
function register_texture!(collector::TextureCollector, tex::Texture{T}) where T
    # Group by the concrete type of the data matrix
    DataMatrixType = typeof(tex.data)

    # Get or create slot for this data type
    slot = get(collector.type_to_slot, DataMatrixType, UInt8(0))
    if slot == UInt8(0)
        slot = UInt8(length(collector.type_order) + 1)
        collector.type_to_slot[DataMatrixType] = slot
        push!(collector.type_order, DataMatrixType)
        collector.data_by_type[DataMatrixType] = Any[]
    end

    # Add texture DATA (not the Texture struct) and get index
    data_vec = collector.data_by_type[DataMatrixType]
    push!(data_vec, tex.data)
    idx = UInt32(length(data_vec))

    return (slot, idx)
end

"""
    build_texture_tuple(collector::TextureCollector) -> Tuple

Build a tuple of texture data vectors from collected textures.
Each slot contains a Vector of data matrices of the same concrete type.
These can be converted to device arrays for GPU rendering.
"""
function build_texture_tuple(collector::TextureCollector)
    if isempty(collector.type_order)
        return ()
    end

    # Build typed vectors of DATA matrices
    vectors = []
    for DT in collector.type_order
        data_mats = collector.data_by_type[DT]
        # Create properly typed vector of matrices
        typed_vec = Vector{DT}(undef, length(data_mats))
        for (i, data) in enumerate(data_mats)
            typed_vec[i] = data
        end
        push!(vectors, typed_vec)
    end

    return Tuple(vectors)
end

# ============================================================================
# Texture to TextureRef Conversion
# ============================================================================

"""
    texture_to_ref(tex::Texture{T}, collector::TextureCollector) -> TextureRef{T}

Convert a Texture to TextureRef, registering non-constant textures in the collector.
"""
function texture_to_ref(tex::Texture{T}, collector::TextureCollector) where T
    if tex.isconst
        # Constant texture - no need to store in tuple
        return TextureRef(tex.const_value)
    else
        # Register texture and get index
        slot, idx = register_texture!(collector, tex)
        return TextureRef{T}(slot, idx)
    end
end

# Passthrough for already-converted TextureRef
texture_to_ref(ref::TextureRef, ::TextureCollector) = ref
