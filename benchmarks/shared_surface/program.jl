# A finite capability mask and capacity replace the exact ordered node tuple.
const SURFACE_SIMPLE_MASK = foldl(|, (UInt32(1) << k for k in (1, 2, 3, 4)); init = UInt32(0))
const SURFACE_PRINCIPLED_MASK = foldl(|, (UInt32(1) << k for k in (1, 2, 3, 4, 5, 6, 7, 10)); init = UInt32(0))
@inline surface_has_walk(::Val{Mask}) where {Mask} = Mask isa Bool ? Mask : !iszero(Mask & (UInt32(1) << 14))
@inline function surface_apply(f, l::SurfaceStoredLobe{Looped, Mask}, args...) where {Looped, Mask}
    return surface_apply_fields(f, Val(Looped), Val(Mask), l.kind, l.data..., args...)
end

# Pass the record fields as values. Passing a pointer to a record inside the
# caller's packed array would require a different Function-storage pointee type.
Base.@constprop :none @noinline @generated function surface_apply_fields(
        f, ::Val{Looped}, ::Val{Mask}, kind::UInt32,
        d1::Float32, d2::Float32, d3::Float32, d4::Float32, d5::Float32,
        d6::Float32, d7::Float32, d8::Float32, d9::Float32, d10::Float32,
        d11::Float32, d12::Float32, d13::Float32, d14::Float32, d15::Float32, args...
    ) where {Looped, Mask}
    kinds = Mask isa Bool ? [1:13;15:20] : [k for k in 1:20 if !iszero(Mask & (UInt32(1) << k))]
    Mask === true && push!(kinds, 14)
    # The host classifier and surface_store establish this closed tag set.
    # Zero-weight padding is skipped before dispatch. An unreachable error here
    # makes GPUCompiler inline the whole evaluator to propagate invocation aborts.
    body = :(f(surface_restore(l, Val($(last(kinds)))), args...))
    for kind in reverse(kinds[1:(end - 1)])
        body = :(l.kind == $kind ? f(surface_restore(l, Val($kind)), args...) : $body)
    end
    return quote
        $(Expr(:meta, :noinline))
        l = SurfaceStoredLobe{Looped, Mask}(
            kind, (
                d1, d2, d3, d4, d5, d6, d7, d8, d9, d10,
                d11, d12, d13, d14, d15,
            )
        )
        $body
    end
end
subsurface_slot(::Type{L}) where {Looped, Mask, L <: Tuple{SurfaceStoredWeight{Looped, Mask}, Vararg{SurfaceStoredWeight{Looped, Mask}}}} =
    surface_has_walk(Val(Mask)) ? fieldcount(L) : 0

struct SurfaceProgramNodes{N, Looped, Mask, Emissive} end
emits(::Type{SurfaceProgramNodes{N, L, M, E}}) where {N, L, M, E} = E
surface_node_mask(::Type) = nothing
surface_node_mask(::Type{<:PrincipledBSDF{T, S, E}}) where {T, S, E} =
    SURFACE_PRINCIPLED_MASK | (S ? UInt32(1) << 14 : UInt32(0))
surface_node_mask(::Type{<:Union{DiffuseBSDF, SheenBSDF, GlossyBSDF, MetallicBSDF, MetallicFilmBSDF, TransparentBSDF, EmissionBSDF}}) = SURFACE_SIMPLE_MASK
surface_node_mask(::Type{<:PhysicalMetallicBSDF}) = SURFACE_SIMPLE_MASK | (UInt32(1) << 8)
surface_node_mask(::Type{<:TranslucentBSDF}) = SURFACE_SIMPLE_MASK | (UInt32(1) << 11)
surface_node_mask(::Type{ToonBSDF{G}}) where {G} = SURFACE_SIMPLE_MASK | (UInt32(1) << (G ? 13 : 12))
surface_node_mask(::Type{SubsurfaceBSDF{S}}) where {S} = SURFACE_SIMPLE_MASK | (S ? UInt32(1) << 14 : UInt32(0))
surface_node_mask(::Type{RefractionBSDF{C}}) where {C} = SURFACE_SIMPLE_MASK | (UInt32(1) << (C === CyclesBeckmann ? 20 : 9))
surface_beckmann_bit(::Type{<:PrincipledBSDF}) = UInt32(1) << 18
surface_beckmann_bit(::Type{GlossyBSDF}) = UInt32(1) << 15
surface_beckmann_bit(::Type{<:Union{MetallicBSDF, MetallicFilmBSDF}}) = UInt32(1) << 16
surface_beckmann_bit(::Type{<:PhysicalMetallicBSDF}) = UInt32(1) << 19
function surface_node_mask(::Type{SingleScatterBSDF{B, C}}) where {B, C}
    mask = surface_node_mask(B)
    mask === nothing && return nothing
    # Stable distribution capability, independent of node count and order.
    if C === CyclesBeckmann
        mask |= surface_beckmann_bit(B)
        B <: PrincipledBSDF && (mask |= (UInt32(1) << 16) | (UInt32(1) << 17))
    end
    return mask
end
surface_node_capacity(::Type, mask) = 1
surface_node_capacity(::Type{<:Union{TransparentBSDF, EmissionBSDF}}, mask) = 0
surface_node_capacity(::Type{<:PrincipledBSDF{T}}, mask) where {T} =
    (T ? 7 : 6) + (surface_has_walk(Val(mask)) ? 2 : 0)
surface_node_capacity(::Type{SingleScatterBSDF{B, C}}, mask) where {B, C} = surface_node_capacity(B, mask)
surface_node_capacity(::Type{<:SubsurfaceBSDF}, mask) = surface_has_walk(Val(mask)) ? 2 : 1

function surface_program_class(::Type{M}, api) where {M <: Tuple}
    masks = map(surface_node_mask, M.parameters)
    any(isnothing, masks) && return nothing
    mask = foldl(|, masks; init = SURFACE_SIMPLE_MASK)
    slots = sum(T -> surface_node_capacity(T, mask), M.parameters; init = 0)
    slots > 32 && return nothing
    n = slots == 0 ? 0 : slots <= 4 ? 4 : slots <= 8 ? 8 : slots <= 16 ? 16 : 32
    return SurfaceProgramNodes{n, api isa Mantle.MetalAPI || n > 8, mask, emits(M)}
end
@inline surface_chunks(ins) = ins.op == NodeOp.PRINCIPLED ? instruction_count(PrincipledBSDF) :
    ins.mode[1] == ClosureKind.PHYSICAL_METALLIC_FILM ? instruction_count(PhysicalMetallicBSDF{true}) : 1

@inline function surface_program_transparency(s, prog, sc)
    total_transparency = Vec3f(0);i = Int32(1)
    @inbounds while prog[i].op != NodeOp.RESULT
        ins = prog[i]
        if ins.op == NodeOp.PRINCIPLED
            c = instruction_count(PrincipledBSDF)
            total_transparency += Vec3f(node_f(s, prog, i, 1, c, sc) * (1.0f0 - clamp(node_f(s, prog, i, 6, c, sc), 0.0f0, 1.0f0)))
        elseif ins.mode[1] == ClosureKind.DIFFUSE
            total_transparency += Vec3f(node_f(s, prog, i, 1, 1, sc) * (1.0f0 - clamp(node_f(s, prog, i, 5, 1, sc), 0.0f0, 1.0f0)))
        elseif ins.mode[1] == ClosureKind.TRANSPARENT
            total_transparency += transparency(s, prog, i, sc)
        end
        i = next_node(prog, i, surface_chunks(ins))
    end
    return total_transparency
end
@inline function surface_zero_prefix(s, prog, i, sc)
    emission = Vec3f(0)
    @inbounds while prog[i].op == NodeOp.CLOSURE && prog[i].mode[1] in (ClosureKind.TRANSPARENT, ClosureKind.EMISSION)
        if prog[i].mode[1] == ClosureKind.EMISSION
            _, _, e = node_lobes(EmissionBSDF, s, prog, i, sc, nothing)
            emission += e
        end
        i = next_node(prog, i, 1)
    end
    return i, emission
end
@inline surface_empty(::Val{N}, ::Val{Looped}, ::Val{Mask}) where {N, Looped, Mask} =
    ntuple(_ -> Weighted(SurfaceStoredLobe{Looped, Mask}(0, ntuple(_ -> 0.0f0, Val(15))), SpectralRadiance(0.0f0), 0.0f0, Vec3f(0, 0, 1), Vec3f(0)), Val(N))

# Set up each node once into a local mutable array, then expose an isbits tuple
# to the existing BSDF carrier. This avoids recursively expanding every possible
# node kind at every remaining capacity during inference/LLVM optimization.
@inline surface_append!(storage, ::Tuple{}, at, table, lambda, scale, mode, mask) = at
@inline function surface_append!(storage, terms::Tuple, at, table, lambda, scale, mode, mask)
    @inbounds storage[at] = surface_store(lift(first(terms), scale, table, lambda), mode, mask)
    return surface_append!(storage, Base.tail(terms), at + Int32(1), table, lambda, scale, mode, mask)
end
@inline function surface_store_node!(::Type{T}, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask) where {T}
    terms, _, emission = node_lobes(T, s, prog, i, sc, lut)
    after = surface_append!(storage, terms, at, table, lambda, scale, mode, mask)
    return next_node(prog, i, instruction_count(T)), after, emission
end
Base.@constprop :none @noinline function surface_setup_slots(::Val{N}, s, prog, i, sc, lut, table, lambda, scale, mode::Val{Looped}, mask::Val{Mask}) where {N, Looped, Mask}
    storage = MVector{N, SurfaceStoredWeight{Looped, Mask}}(surface_empty(Val(N), mode, mask))
    at = Int32(1);emission = Vec3f(0)
    @inbounds while prog[i].op != NodeOp.RESULT
        i, prefix = surface_zero_prefix(s, prog, i, sc)
        emission += prefix
        prog[i].op == NodeOp.RESULT && break
        i, at, e = surface_store_dispatch!(storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
        emission += e
    end
    return Tuple(storage), emission
end
@inline function surface_principled_store!(storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask::Val{Mask}) where {Mask}
    p = principled_inputs(s, prog, i, sc)
    @inbounds distribution = prog[i].mode[2]
    if !iszero(Mask & (UInt32(1) << 18)) && distribution == 0x02
        after, emission = surface_principled_layers!(
            storage, at, p, sc, lut, table, lambda, scale,
            mode, mask, false, CyclesBeckmann()
        )
    else
        after, emission = surface_principled_layers!(
            storage, at, p, sc, lut, table, lambda, scale,
            mode, mask, distribution == 0x00, CyclesGGX()
        )
    end
    return next_node(prog, i, instruction_count(PrincipledBSDF)), after, emission
end
@inline function surface_principled_layers!(
        storage, at, p, sc, lut, table, lambda, scale, mode, mask,
        multi, convention
    )
    ss = surface_has_walk(mask)
    layer = principled_base_layers(Val(ss), p, sc, lut, multi, convention)
    at = surface_append!(
        storage, map(kept, (layer.l_sheen, layer.l_coat, layer.l_metal)),
        at, table, lambda, scale, mode, mask
    )
    (; transmission, w, vn, roughness, ior, specular_tint, base_clamped, film_thickness, film_ior) = layer
    thin = p.thin_wall > 0.5f0
    if thin
        terms, w = principled_transmission(
            Val(true), transmission, w, vn, roughness, ior, specular_tint,
            base_clamped, film_thickness, film_ior, p, sc, lut, multi, convention
        )
        at = surface_append!(storage, map(kept, terms), at, table, lambda, scale, mode, mask)
    else
        terms, w = principled_transmission(
            Val(false), transmission, w, vn, roughness, ior, specular_tint,
            base_clamped, film_thickness, film_ior, p, sc, lut, multi, convention
        )
        at = surface_append!(storage, map(kept, terms), at, table, lambda, scale, mode, mask)
    end
    specular, w = principled_specular_layer(p, layer, w, lut, multi, convention)
    at = surface_append!(storage, (kept(specular),), at, table, lambda, scale, mode, mask)
    if ss
        if thin
            terms = principled_subsurface(
                Val(true), Val(true), layer.subsurface, w, layer.n, roughness,
                ior, layer.diffuse_roughness, base_clamped, p, layer.wo
            )
            at = surface_append!(storage, map(kept, terms), at, table, lambda, scale, mode, mask)
        else
            terms = principled_subsurface(
                Val(false), Val(true), layer.subsurface, w, vn, roughness,
                ior, layer.diffuse_roughness, base_clamped, p, layer.wo
            )
            at = surface_append!(storage, map(kept, terms), at, table, lambda, scale, mode, mask)
        end
    end
    diffuse = principled_diffuse_layer(layer, w)
    at = surface_append!(storage, (kept(diffuse),), at, table, lambda, scale, mode, mask)
    return at, layer.emission
end
@inline function surface_distributed_store!(::Type{T}, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask::Val{Mask}) where {T, Mask}
    @inbounds distribution = prog[i].mode[2]
    if !iszero(Mask & surface_beckmann_bit(T)) && distribution == 0x02
        return surface_store_node!(SingleScatterBSDF{T, CyclesBeckmann}, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    end
    terms, _, emission = surface_ggx_node_lobes(T, s, prog, i, sc, lut, distribution == 0x00)
    after = surface_append!(storage, terms, at, table, lambda, scale, mode, mask)
    return next_node(prog, i, instruction_count(T)), after, emission
end

# GGX single/multiple scattering share the same lobe layout. A runtime energy
# switch avoids cloning the entire node setup for each scattering choice.
@inline surface_ggx_node_lobes(::Type{PrincipledBSDF{T, S, E}}, s, prog, i, sc, lut, multi) where {T, S, E} =
    principled_lobes(Val(T), Val(S), principled_inputs(s, prog, i, sc), sc, lut, multi)
@inline surface_ggx_node_lobes(::Type{GlossyBSDF}, s, prog, i, sc, lut, multi) =
    reflection_node(Val(false), s, prog, i, sc, lut, Val(false), CyclesGGX(), multi)
@inline surface_ggx_node_lobes(::Type{MetallicBSDF}, s, prog, i, sc, lut, multi) =
    reflection_node(Val(true), s, prog, i, sc, lut, Val(false), CyclesGGX(), multi)
@inline surface_ggx_node_lobes(::Type{MetallicFilmBSDF}, s, prog, i, sc, lut, multi) =
    reflection_node(Val(true), s, prog, i, sc, lut, Val(true), CyclesGGX(), multi)
@inline surface_ggx_node_lobes(::Type{PhysicalMetallicBSDF{Film}}, s, prog, i, sc, lut, multi) where {Film} =
    physical_metallic_node(Val(Film), s, prog, i, sc, lut, CyclesGGX(), multi)

@inline function surface_store_dispatch!(storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask::Val{Mask}) where {Mask}
    @inbounds ins = prog[i]
    if !iszero(Mask & (UInt32(1) << 7))
        ins.op == NodeOp.PRINCIPLED && return surface_principled_store!(storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    end
    kind = ins.mode[1]
    kind == ClosureKind.DIFFUSE && return surface_store_node!(DiffuseBSDF, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    kind == ClosureKind.SHEEN && return surface_store_node!(SheenBSDF, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    kind == ClosureKind.GLOSSY && return surface_distributed_store!(GlossyBSDF, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    kind == ClosureKind.METALLIC && return surface_distributed_store!(MetallicBSDF, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    kind == ClosureKind.METALLIC_FILM && return surface_distributed_store!(MetallicFilmBSDF, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    if !iszero(Mask & (UInt32(1) << 8))
        kind == ClosureKind.PHYSICAL_METALLIC && return surface_distributed_store!(PhysicalMetallicBSDF{false}, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
        kind == ClosureKind.PHYSICAL_METALLIC_FILM && return surface_distributed_store!(PhysicalMetallicBSDF{true}, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    end
    if !iszero(Mask & (UInt32(1) << 11))
        kind == ClosureKind.TRANSLUCENT && return surface_store_node!(TranslucentBSDF, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    end
    if !iszero(Mask & (UInt32(1) << 12))
        kind == ClosureKind.TOON_DIFFUSE && return surface_store_node!(ToonBSDF{false}, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    end
    if !iszero(Mask & (UInt32(1) << 13))
        kind == ClosureKind.TOON_GLOSSY && return surface_store_node!(ToonBSDF{true}, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    end
    if !iszero(Mask & (UInt32(1) << 9))
        kind == ClosureKind.REFRACTION_GGX && return surface_store_node!(RefractionBSDF{CyclesGGX}, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    end
    if !iszero(Mask & (UInt32(1) << 20))
        kind == ClosureKind.REFRACTION_BECKMANN && return surface_store_node!(RefractionBSDF{CyclesBeckmann}, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    end
    # surface_program_class accepts only the enumerated program-producing nodes.
    # After the branches above, its only remaining node is SubsurfaceBSDF.
    return surface_store_node!(SubsurfaceBSDF{surface_has_walk(mask)}, storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
end
@inline function surface_program_bsdf(::Type{SurfaceProgramNodes{N, Looped, Mask, E}}, s, prog, sc, lut, table, lambda) where {N, Looped, Mask, E}
    alpha = 1.0f0 - _avg(surface_program_transparency(s, prog, sc))
    scale = alpha > 1.0f-6 ? 1.0f0 / alpha : 0.0f0
    slots, emission = surface_setup_slots(Val(N), s, prog, Int32(1), sc, lut, table, lambda, scale, Val(Looped), Val(Mask))
    total = reduce(+, map(w -> w.sample_weight, slots); init = 0.0f0)
    L = E && _max(emission) > 0.0f0 ? uplift_rgb_illuminant(table, RGBSpectrum(emission...), lambda) / D65_PHOTOMETRIC : SpectralRadiance(0.0f0)
    return LobeBSDF(slots, total, lambda, L, disperses(slots))
end
@inline function material_bxdf(
        mat::GatheredShader{M}, table, textures, work, camera, samples_per_pixel::Int32,
        lambda::Wavelengths, regularize::Bool
    ) where {M <: SurfaceProgramNodes}
    program = @inbounds mat.inputs.program[mat.program_offset + work.material_idx.vec_idx]
    prog = closure_programs(textures)[program]
    s = GatheredStack(mat.inputs.values, mat.inputs.capacity, mat.hit)
    return surface_program_bsdf(M, s, prog, closure_context(work), LobeSetup(lookup_tables(textures), regularize), table, lambda)
end
