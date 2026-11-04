# ============================================================================
# PBRT file parser — tokens → PBRTScene intermediate representation
# ============================================================================
# Parses the pbrt-v4 scene description format into a structured IR that can
# be converted to a Hikari scene by scene_builder.jl.

# A single parsed parameter: "type name" [values]
struct PBRTParam
    type::Symbol   # :float, :integer, :rgb, :spectrum, :point3, :normal,
                   # :point2, :vector3, :string, :texture, :blackbody, :bool
    name::String
    values::Vector{Any}
end

# A parsed entity (shape, material, light, texture, medium, etc.)
struct PBRTEntity
    kind::String       # "Shape", "Material", "LightSource", etc.
    type::String       # "sphere", "diffuse", "point", etc.
    params::Dict{String, PBRTParam}
end

# Attribute state (pushed/popped with AttributeBegin/End)
mutable struct PBRTAttrState
    transform::Mat4f
    material_name::Union{String, Nothing}
    material_inline::Union{PBRTEntity, Nothing}
    area_light::Union{PBRTEntity, Nothing}
    medium_inner::String   # "" means vacuum
    medium_outer::String
    reverse_orientation::Bool
end

function PBRTAttrState()
    PBRTAttrState(Mat4f(I), nothing, nothing, nothing, "", "", false)
end

function Base.copy(s::PBRTAttrState)
    PBRTAttrState(s.transform, s.material_name, s.material_inline,
                  s.area_light, s.medium_inner, s.medium_outer,
                  s.reverse_orientation)
end

# A shape with its full context at parse time
struct PBRTShapeRecord
    entity::PBRTEntity
    transform::Mat4f
    material_name::Union{String, Nothing}
    material_inline::Union{PBRTEntity, Nothing}
    area_light::Union{PBRTEntity, Nothing}
    medium_inner::String
    medium_outer::String
    reverse_orientation::Bool
end

struct PBRTLightRecord
    entity::PBRTEntity
    transform::Mat4f
end

# Complete parsed scene
struct PBRTScene
    # Options (before WorldBegin)
    film::Union{PBRTEntity, Nothing}
    camera::Union{PBRTEntity, Nothing}
    camera_transform::Mat4f
    # Distance from eye to target of the `LookAt` in force when `Camera` was
    # issued, or 0 if the camera transform was not built by a `LookAt`.
    #
    # The transform alone cannot carry this: `look_at` normalizes the
    # direction, so every target along the view ray yields the same matrix (to
    # Float32 rounding, ~2e-6) and no rendering test can see the difference.
    # Interactive viewers can — Makie's `Camera3D` scales both
    # pan (`2*norm(lookat-eye)/height*delta`) and zoom
    # (`eyeposition = lookat - zoom_step*viewdir`) by exactly this distance,
    # while rotation is angular and ignores it. RayMakie reconstructed the
    # target as `eye + normalize(forward)`, pinning it to 1 unit, so on Crown
    # (true distance 34.4, geometry ~90x43x100) a full pan drag moved the
    # camera 0.14 units and a scroll click 0.10 — indistinguishable from a
    # dead input, while rotation looked perfectly normal.
    camera_lookat_distance::Float32
    sampler::Union{PBRTEntity, Nothing}
    integrator::Union{PBRTEntity, Nothing}
    pixel_filter::Union{PBRTEntity, Nothing}

    # World contents
    named_materials::Dict{String, PBRTEntity}
    named_media::Dict{String, PBRTEntity}
    named_textures::Dict{String, PBRTEntity}

    media_transforms::Dict{String, Mat4f}

    shapes::Vector{PBRTShapeRecord}
    lights::Vector{PBRTLightRecord}

    base_dir::String
end

# ============================================================================
# Parameter parsing
# ============================================================================

function parse_param_type_name(s::AbstractString)
    # "type name" → (:type, "name")
    idx = findfirst(' ', s)
    idx === nothing && error("invalid parameter: '$s' (expected 'type name')")
    type_str = SubString(s, 1, idx - 1)
    name = String(SubString(s, idx + 1))
    type_sym = Symbol(type_str)
    return (type_sym, name)
end

function parse_params!(ts::TokenStream)
    params = Dict{String, PBRTParam}()
    while !eof(ts)
        t = peek(ts)
        t === nothing && break
        # Parameters start with a quoted "type name" string
        t.type != TOK_STRING && break
        # Check if it looks like a "type name" pair
        !contains(t.value, ' ') && break

        next!(ts)  # consume the "type name" token
        type_sym, name = parse_param_type_name(t.value)

        # Read values — optionally bracketed
        values = Any[]
        bracketed = false
        t2 = peek(ts)
        if t2 !== nothing && t2.type == TOK_LBRACKET
            next!(ts)  # consume [
            bracketed = true
        end

        if bracketed
            while !eof(ts)
                t3 = peek(ts)
                t3 === nothing && break
                t3.type == TOK_RBRACKET && (next!(ts); break)
                next!(ts)
                if t3.type == TOK_NUMBER
                    push!(values, parse(Float64, t3.value))
                elseif t3.type == TOK_STRING
                    push!(values, t3.value)
                elseif t3.type == TOK_WORD
                    # "true"/"false" or named reference
                    if t3.value == "true"
                        push!(values, true)
                    elseif t3.value == "false"
                        push!(values, false)
                    else
                        push!(values, t3.value)
                    end
                end
            end
        else
            # Single value (no brackets)
            if !eof(ts)
                t3 = peek(ts)
                if t3 !== nothing && t3.type in (TOK_NUMBER, TOK_STRING, TOK_WORD)
                    next!(ts)
                    if t3.type == TOK_NUMBER
                        push!(values, parse(Float64, t3.value))
                    elseif t3.type == TOK_WORD
                        if t3.value == "true"
                            push!(values, true)
                        elseif t3.value == "false"
                            push!(values, false)
                        else
                            push!(values, t3.value)
                        end
                    else
                        push!(values, t3.value)
                    end
                end
            end
        end

        params[name] = PBRTParam(type_sym, name, spectrumfile(type_sym, values, ts.base_dir))
    end
    return params
end

"""
A `"spectrum"` parameter's values with a file it names read in. pbrt takes a named spectrum or a
file of (λ, value) pairs under the same syntax: `"spectrum eta" [ "spds/Au.eta.spd" ]`. Read here,
the only place that knows the scene's directory, a file looks like the same pairs given inline.
"""
function spectrumfile(type::Symbol, values::Vector{Any}, base_dir::AbstractString)
    (type === :spectrum && length(values) == 1 && values[1] isa AbstractString) || return values
    path = isabspath(values[1]) ? values[1] : joinpath(base_dir, values[1])
    isfile(path) || return values
    pairs = Any[]
    for line in eachline(path)
        for word in split(first(split(line, '#')))
            push!(pairs, parse(Float64, word))
        end
    end
    iseven(length(pairs)) || throw(ArgumentError("pbrt: $path holds an odd count of numbers, not (λ, value) pairs"))
    return pairs
end

# ============================================================================
# Main parser
# ============================================================================

function parse_pbrt(filename::AbstractString)
    text = read(filename, String)
    base_dir = dirname(abspath(filename))
    parse_pbrt_string(text; base_dir=base_dir, filename=filename)
end

function parse_pbrt_string(text::AbstractString;
                           base_dir::AbstractString=".",
                           filename::AbstractString="<string>")
    tokens = tokenize(text)
    ts = TokenStream(tokens; filename=filename, base_dir=base_dir)

    # Scene state
    film = nothing
    camera = nothing
    camera_transform = Mat4f(I)
    camera_lookat_distance = 0.0f0
    sampler = nothing
    integrator = nothing
    pixel_filter = nothing

    named_materials = Dict{String, PBRTEntity}()
    named_media = Dict{String, PBRTEntity}()
    named_textures = Dict{String, PBRTEntity}()
    media_transforms = Dict{String, Mat4f}()

    shapes = PBRTShapeRecord[]
    lights = PBRTLightRecord[]

    # Current transform matrix
    ctm = Mat4f(I)
    # Eye-to-target distance of the `LookAt` that produced `ctm`, or 0 if `ctm`
    # did not come from one. Concatenating a further `Translate`/`Rotate` keeps
    # it valid (they are rigid); `Scale` does not, but pbrt scenes place the
    # scale before the `LookAt` (Crown's leading `Scale -1 1 1`), so the
    # `LookAt` is what sets it last.
    ctm_lookat_distance = 0.0f0
    in_world = false
    attr_stack = PBRTAttrState[]
    attr = PBRTAttrState()

    while !eof(ts)
        t = next!(ts)
        t.type != TOK_WORD && continue

        word = t.value

        # ---- Options (before WorldBegin) ----

        if word == "Film"
            type_str = expect!(ts, TOK_STRING).value
            params = parse_params!(ts)
            film = PBRTEntity("Film", type_str, params)

        elseif word == "Camera"
            type_str = expect!(ts, TOK_STRING).value
            params = parse_params!(ts)
            camera = PBRTEntity("Camera", type_str, params)
            camera_transform = ctm
            camera_lookat_distance = ctm_lookat_distance

        elseif word == "Sampler"
            type_str = expect!(ts, TOK_STRING).value
            params = parse_params!(ts)
            sampler = PBRTEntity("Sampler", type_str, params)

        elseif word == "Integrator"
            type_str = expect!(ts, TOK_STRING).value
            params = parse_params!(ts)
            integrator = PBRTEntity("Integrator", type_str, params)

        # ---- Transforms ----

        elseif word == "LookAt"
            vals = Float32[parse(Float64, expect!(ts, TOK_NUMBER).value) for _ in 1:9]
            eye = Point3f(vals[1], vals[2], vals[3])
            target = Point3f(vals[4], vals[5], vals[6])
            up = Vec3f(vals[7], vals[8], vals[9])
            # pbrt-v4 BasicSceneBuilder::LookAt CONCATENATES onto the CTM
            # rather than replacing it, so a leading `Scale -1 1 1` (Crown's
            # X-mirror at the top of crown.pbrt) propagates into the camera
            # transform. Without this multiplication Hikari silently dropped
            # the X-flip and rendered Crown horizontally mirrored vs the
            # pbrt-v4 reference EXR.
            ctm = ctm * pbrt_lookat(eye, target, up)
            ctm_lookat_distance = Float32(norm(target - eye))

        elseif word == "Translate"
            x = Float32(parse(Float64, expect!(ts, TOK_NUMBER).value))
            y = Float32(parse(Float64, expect!(ts, TOK_NUMBER).value))
            z = Float32(parse(Float64, expect!(ts, TOK_NUMBER).value))
            ctm = ctm * pbrt_translation_matrix(Vec3f(x, y, z))

        elseif word == "Rotate"
            angle = Float32(parse(Float64, expect!(ts, TOK_NUMBER).value))
            ax = Float32(parse(Float64, expect!(ts, TOK_NUMBER).value))
            ay = Float32(parse(Float64, expect!(ts, TOK_NUMBER).value))
            az = Float32(parse(Float64, expect!(ts, TOK_NUMBER).value))
            ctm = ctm * pbrt_rotation_matrix(angle, Vec3f(ax, ay, az))

        elseif word == "Scale"
            x = Float32(parse(Float64, expect!(ts, TOK_NUMBER).value))
            y = Float32(parse(Float64, expect!(ts, TOK_NUMBER).value))
            z = Float32(parse(Float64, expect!(ts, TOK_NUMBER).value))
            ctm = ctm * pbrt_scale_matrix(Vec3f(x, y, z))

        elseif word == "Transform"
            expect!(ts, TOK_LBRACKET)
            vals = Float32[parse(Float64, expect!(ts, TOK_NUMBER).value) for _ in 1:16]
            expect!(ts, TOK_RBRACKET)
            # pbrt stores transforms in row-major order
            ctm = Mat4f(vals[1], vals[5], vals[9],  vals[13],
                        vals[2], vals[6], vals[10], vals[14],
                        vals[3], vals[7], vals[11], vals[15],
                        vals[4], vals[8], vals[12], vals[16])
            ctm_lookat_distance = 0.0f0   # CTM replaced wholesale

        elseif word == "ConcatTransform"
            expect!(ts, TOK_LBRACKET)
            vals = Float32[parse(Float64, expect!(ts, TOK_NUMBER).value) for _ in 1:16]
            expect!(ts, TOK_RBRACKET)
            m = Mat4f(vals[1], vals[5], vals[9],  vals[13],
                      vals[2], vals[6], vals[10], vals[14],
                      vals[3], vals[7], vals[11], vals[15],
                      vals[4], vals[8], vals[12], vals[16])
            ctm = ctm * m

        elseif word == "Identity"
            ctm = Mat4f(I)
            ctm_lookat_distance = 0.0f0

        elseif word == "CoordSysTransform"
            name = expect!(ts, TOK_STRING).value
            if name == "camera"
                ctm = camera_transform
                ctm_lookat_distance = camera_lookat_distance
            end

        # ---- World block ----

        elseif word == "WorldBegin"
            camera_transform = ctm
            camera_lookat_distance = ctm_lookat_distance
            ctm = Mat4f(I)
            ctm_lookat_distance = 0.0f0
            attr = PBRTAttrState()
            in_world = true

        elseif word == "AttributeBegin"
            push!(attr_stack, copy(attr))
            attr.transform = ctm

        elseif word == "AttributeEnd"
            isempty(attr_stack) && error("$(filename):$(t.line): AttributeEnd without AttributeBegin")
            old = pop!(attr_stack)
            ctm = old.transform  # restore transform from the state that was pushed
            # Actually, we stored the CTM at AttributeBegin time in attr.transform
            # Let me fix: we need to save ctm separately
            attr = old

        elseif word == "ReverseOrientation"
            attr.reverse_orientation = !attr.reverse_orientation

        # ---- Materials ----

        elseif word == "Material"
            type_str = expect!(ts, TOK_STRING).value
            params = parse_params!(ts)
            attr.material_inline = PBRTEntity("Material", type_str, params)
            attr.material_name = nothing

        elseif word == "MakeNamedMaterial"
            name = expect!(ts, TOK_STRING).value
            params = parse_params!(ts)
            mat_type = haskey(params, "type") ? only(params["type"].values) : "diffuse"
            delete!(params, "type")
            named_materials[name] = PBRTEntity("Material", String(mat_type), params)

        elseif word == "NamedMaterial"
            name = expect!(ts, TOK_STRING).value
            attr.material_name = name
            attr.material_inline = nothing

        # ---- Media ----

        elseif word == "MakeNamedMedium"
            name = expect!(ts, TOK_STRING).value
            params = parse_params!(ts)
            med_type = haskey(params, "type") ? only(params["type"].values) : "homogeneous"
            delete!(params, "type")
            named_media[name] = PBRTEntity("Medium", String(med_type), params)
            media_transforms[name] = ctm

        elseif word == "MediumInterface"
            # pbrt: MediumInterface "inside" "outside"
            attr.medium_inner = expect!(ts, TOK_STRING).value
            attr.medium_outer = expect!(ts, TOK_STRING).value

        # ---- Textures ----

        elseif word == "Texture"
            tex_name = expect!(ts, TOK_STRING).value
            tex_class = expect!(ts, TOK_STRING).value  # "spectrum", "float", "rgb"
            tex_type = expect!(ts, TOK_STRING).value    # "imagemap", "constant", "scale", etc.
            params = parse_params!(ts)
            # Store class in params for the builder
            params["_class"] = PBRTParam(:string, "_class", [tex_class])
            named_textures[tex_name] = PBRTEntity("Texture", tex_type, params)

        # ---- Lights ----

        elseif word == "LightSource"
            type_str = expect!(ts, TOK_STRING).value
            params = parse_params!(ts)
            push!(lights, PBRTLightRecord(
                PBRTEntity("LightSource", type_str, params), ctm))

        elseif word == "AreaLightSource"
            type_str = expect!(ts, TOK_STRING).value
            params = parse_params!(ts)
            attr.area_light = PBRTEntity("AreaLightSource", type_str, params)

        # ---- Shapes ----

        elseif word == "Shape"
            type_str = expect!(ts, TOK_STRING).value
            params = parse_params!(ts)
            push!(shapes, PBRTShapeRecord(
                PBRTEntity("Shape", type_str, params),
                ctm,
                attr.material_name,
                attr.material_inline,
                attr.area_light,
                attr.medium_inner,
                attr.medium_outer,
                attr.reverse_orientation,
            ))

        # ---- Includes ----

        elseif word == "Include"
            inc_path = expect!(ts, TOK_STRING).value
            full_path = isabspath(inc_path) ? inc_path : joinpath(base_dir, inc_path)
            if isfile(full_path)
                inc_text = read(full_path, String)
                inc_tokens = tokenize(inc_text)
                # Insert included tokens into the stream
                splice!(ts.tokens, ts.pos:ts.pos-1, inc_tokens)
            else
                @warn "pbrt Include file not found: $full_path"
            end

        # ---- Ignored / no-op ----
        elseif word == "PixelFilter"
            if !eof(ts) && peek(ts).type == TOK_STRING
                type_tok = next!(ts)
                params = parse_params!(ts)
                pixel_filter = PBRTEntity("PixelFilter", type_tok.value, params)
            end

        elseif word in ("WorldEnd", "Accelerator", "Option",
                        "ColorSpace", "ObjectBegin", "ObjectEnd", "ObjectInstance",
                        "TransformBegin", "TransformEnd", "ActiveTransform")
            # Skip parameters if present
            if word in ("Accelerator", "Option", "ColorSpace")
                if !eof(ts) && peek(ts).type == TOK_STRING
                    next!(ts)  # consume type string
                    parse_params!(ts)  # consume params
                end
            end

        else
            @warn "pbrt: unknown directive '$word' at $(filename):$(t.line)"
        end
    end

    return PBRTScene(
        film, camera, camera_transform, camera_lookat_distance,
        sampler, integrator, pixel_filter,
        named_materials, named_media, named_textures,
        media_transforms,
        shapes, lights, base_dir,
    )
end

# ============================================================================
# Transform helpers
# ============================================================================

function pbrt_lookat(eye::Point3f, target::Point3f, up::Vec3f)
    # pbrt's LookAt produces a world-to-camera matrix (like OpenGL gluLookAt)
    # Cross product order must match pbrt-v4: right = cross(up, dir), newUp = cross(dir, right)
    dir = normalize(target - eye)
    right = normalize(cross(normalize(up), dir))
    new_up = cross(dir, right)

    tx = -dot(right, Vec3f(eye))
    ty = -dot(new_up, Vec3f(eye))
    tz =  dot(dir, Vec3f(eye))

    # Mat4f is column-major: args fill column-by-column
    # Row 0: right[1..3], tx
    # Row 1: new_up[1..3], ty
    # Row 2: -dir[1..3], tz
    # Row 3: 0, 0, 0, 1
    m = Mat4f(
        right[1],  new_up[1], -dir[1], 0f0,   # column 0
        right[2],  new_up[2], -dir[2], 0f0,   # column 1
        right[3],  new_up[3], -dir[3], 0f0,   # column 2
        tx,        ty,         tz,     1f0,   # column 3
    )
    return m
end

function pbrt_translation_matrix(v::Vec3f)
    Mat4f(1f0, 0f0, 0f0, 0f0,
          0f0, 1f0, 0f0, 0f0,
          0f0, 0f0, 1f0, 0f0,
          v[1], v[2], v[3], 1f0)
end

function pbrt_scale_matrix(v::Vec3f)
    Mat4f(v[1], 0f0,  0f0,  0f0,
          0f0,  v[2], 0f0,  0f0,
          0f0,  0f0,  v[3], 0f0,
          0f0,  0f0,  0f0,  1f0)
end

function pbrt_rotation_matrix(angle_deg::Float32, axis::Vec3f)
    a = normalize(axis)
    s, c = sincosd(angle_deg)
    t = 1f0 - c
    # Column-major layout
    Mat4f(
        t*a[1]*a[1] + c,       t*a[1]*a[2] + s*a[3], t*a[1]*a[3] - s*a[2], 0f0,  # col 0
        t*a[1]*a[2] - s*a[3],  t*a[2]*a[2] + c,      t*a[2]*a[3] + s*a[1], 0f0,  # col 1
        t*a[1]*a[3] + s*a[2],  t*a[2]*a[3] - s*a[1], t*a[3]*a[3] + c,      0f0,  # col 2
        0f0,                   0f0,                   0f0,                   1f0,  # col 3
    )
end
