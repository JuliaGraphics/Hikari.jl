# Computed shader values and BSDF evaluation in separate kernels.
#
# A shader's kernel compiled its program (the node families it uses) and its BSDF nodes, their
# lighting and sampling together: a node edit that changed the families compiled all of it again,
# ~27k instructions of which the program is ~3k, 13 s a node edit on an RTX 4000 Ada. The values
# kernel (`vp_shader_values_kernel!`) runs the program and stores what the BSDF nodes read of it
# (`gather_inputs!`); the closure kernel of the nodes (`vp_shade_kernel!` over a `ClosureQueue`)
# reads that back (`GatheredStack`) and does the rest. BSDF programs are looked up per material,
# so shaders without computed inputs skip the values pass. A node edit compiles the values
# kernel of the new families alone.

"""
Store the values the BSDF instructions from `prog[i]` on read off the stack `s`, for the hit `hit`,
where the closure kernel finds them: `closure_program` numbers them in this order.
"""
@propagate_inbounds function gather_inputs!(inputs::ShaderInputs, hit::Int32, s, prog, i::Int32)
    g = Int32(0)
    n = length(prog) % Int32
    while i <= n
        ins = prog[i]
        Base.Cartesian.@nexprs 10 a -> begin
            o = ins.arg[a]
            if reads_stack(o)
                inputs.values[g * inputs.capacity + hit] = _get(s, o)
                g += Int32(1)
                if _bit(ins.wide, a)
                    inputs.values[g * inputs.capacity + hit] = _get(s, o + 0x01)
                    inputs.values[(g + Int32(1)) * inputs.capacity + hit] = _get(s, o + 0x02)
                    g += Int32(2)
                end
            end
        end
        ins.op == NodeOp.RESULT && return nothing
        i += Int32(1)
    end
    return nothing
end

"""
The values a closure kernel reads for one hit, by the numbers `closure_program` gave them: what the
values kernel stored (`gather_inputs!`).
"""
struct GatheredStack{V}
    values::V
    capacity::Int32
    hit::Int32
end

@inline _get(s::GatheredStack, o) = @inbounds s.values[(Int32(o) - Int32(1)) * s.capacity + s.hit]

"""
    GatheredShader{M}

The shader a closure kernel of BSDF nodes `M` shades a hit with: the inputs its values kernel
gathered and its closure program.
"""
struct GatheredShader{M, S <: ShaderInputs}
    inputs::S
    hit::Int32
    program_offset::Int32
end

emits(::GatheredShader{M}) where {M} = emits(M)

"""The material a hit is shaded with, of the queue's type or, for a closure kernel, from its gathered inputs."""
@propagate_inbounds shaded_instance(ref::TypedHitRef{T}, materials, key::SetKey) where {T} =
    material_of_type(materials, T, key.vec_idx)
@propagate_inbounds shaded_instance(ref::GatheredRef{M, S}, materials, key::SetKey) where {M, S} =
    GatheredShader{M, S}(ref.inputs, ref.idx, ref.program_offset)

# A shader has no displacement: its normals are its BSDF nodes' inputs.
@propagate_inbounds perturbed_frame(::Type{<:ShaderClosures}, materials, idx::SetKey, ns, dpdus, dpdu, dpdv, ng,
                                    bump_inputs) = (ns, dpdus)

"""The closure programs stored in `textures`, which every shader's are of one type."""
@inline closure_programs(textures) = shader_programs(textures, ShaderProgram{NodeFeature.CLOSURES}(Int32(0)))

# Every coordinate and derivative has already been evaluated by the values
# program. BSDF setup only reads the position and shading frame; constructing a
# camera/filter context here repeats work and keeps it live across helper calls.
@inline function closure_context(work)
    tfc = TextureFilterContext(work.uv, 0f0, 0f0, 0f0, 0f0, work.face_idx, work.bary, work.pi, work.ns)
    sign = dot(cross(work.ns, work.dpdus), work.dpdv) < 0f0 ? -1f0 : 1f0
    return ShaderContext(tfc, work.wo, Vec3f(0), Vec3f(0), work.dpdus, work.n, sign, Vec3f(0), Point2f(0))
end

@propagate_inbounds function material_bxdf(mat::GatheredShader{M}, table, textures, work, camera,
                                           samples_per_pixel::Int32, lambda::Wavelengths,
                                           regularize::Bool) where {M <: Tuple}
    sc = closure_context(work)
    key = work.material_idx
    program = @inbounds mat.inputs.program[mat.program_offset + key.vec_idx]
    prog = closure_programs(textures)[program]
    s = GatheredStack(mat.inputs.values, mat.inputs.capacity, mat.hit)
    produced = node_lobes(M, s, prog, Int32(1), sc, LobeSetup(lookup_tables(textures), regularize))
    return material_bsdf(M, produced, table, lambda)
end

@propagate_inbounds function material_bxdf(mat::GatheredShader{M}, table, textures, work, camera,
                                           samples_per_pixel::Int32, lambda::Wavelengths,
                                           regularize::Bool) where {M <: SurfaceProgramNodes}
    program = @inbounds mat.inputs.program[mat.program_offset + work.material_idx.vec_idx]
    prog = closure_programs(textures)[program]
    s = GatheredStack(mat.inputs.values, mat.inputs.capacity, mat.hit)
    return surface_program_bsdf(M, s, prog, closure_context(work),
                        LobeSetup(lookup_tables(textures), regularize), table, lambda)
end

"""
    vp_shader_values_kernel!(q, surface_hits, cur, arrivals, accel, media_interfaces, materials, inputs, camera, samples_per_pixel)

Run the program of each hit of `q`'s shader type at the hit, and store what its BSDF nodes read of
it in `inputs`, for the closure kernel.
"""
@kernel inbounds = true function vp_shader_values_kernel!(
    q, surface_hits, cur, arrivals, accel, media_interfaces, materials, inputs, camera_ref,
    samples_per_pixel::Int32,
)
    j = @index(Global)
    if j <= queuelength(q)
        vp_shader_values!(q[j], surface_hits, cur, arrivals, accel, media_interfaces, materials, inputs,
                          camera_ref, samples_per_pixel)
    end
end

@propagate_inbounds function vp_shader_values!(ref::TypedHitRef{T}, surface_hits, cur, arrivals, accel,
                                               media_interfaces, materials, inputs::ShaderInputs, camera_ref,
                                               samples_per_pixel::Int32) where {T}
    camera = camera_ref[Int32(1)]
    h = expanded_hit(ref.idx, surface_hits, cur, arrivals, accel, media_interfaces)
    key = h.mi.material
    # The work item the closure kernel shades the hit with: a shader's frame is the geometric one.
    hit_work = VPHitSurfaceWorkItem(
        h.work,
        h.geom.pi, h.geom.ps, h.geom.n, h.geom.dpdu, h.geom.dpdv,
        h.geom.ns, h.geom.dpdus, cross(h.geom.ns, h.geom.dpdus),
        h.geom.uv, key, h.mi,
        h.primitive.metadata.primitive_index, h.bary,
    )
    mat = material_of_type(materials, T, key.vec_idx)
    sc = shader_context(VPMaterialEvalWorkItem(hit_work, h.wo, key), camera, samples_per_pixel,
                        shader_geometry(mat.program, h.geom, h.primitive,
                            () -> bump_filter_context(camera, h.work, h.geom, samples_per_pixel,
                                                      h.primitive.metadata.primitive_index, SVector{3, Float32}(h.bary))))
    prog = shader_programs(materials, mat.program)[mat.program.idx]
    s, i = run_values(prog, materials, sc)
    gather_inputs!(inputs, ref.idx, s, prog, i)
    return nothing
end
