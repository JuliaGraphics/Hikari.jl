"""
Builders for node textures, after Blender's shader nodes, with Blender's names,
settings and defaults. `using Hikari.TextureNodes`, then e.g.

    p = xyz()
    rings = wave(p; type = :rings, scale = 2, distortion = 4, detail = 3).fac
    wood = color_ramp(rings, 0.0 => (0.6, 0.35, 0.15), 0.8 => (0.45, 0.25, 0.1), 1.0 => (0.2, 0.1, 0.04))
    Hikari.Diffuse(Kd = wood)

Inputs are [`uv`](@ref), [`xyz`](@ref) (world-space position) and [`normal`](@ref).
Every socket of a node takes a number, a colour or tuple, or another node's
output. Nodes with several outputs return their first and name the others:
`noise(p).color`, `voronoi(p).position`, `brick(p).fac`.

Arithmetic (`+ - * / ^`, `min`, `max`, `mod` (floored), `rem`, `abs`, `sign`,
`floor`, `ceil`, `round`, `trunc`, `sqrt`, `exp`, `log`, the trigonometric and
hyperbolic functions, `atan(y, x)`, `clamp`) works elementwise on nodes and
plain numbers or colours, like Blender's Math and Vector Math nodes; `dot`,
`cross`, `norm` and `normalize` come from `LinearAlgebra`. [`math`](@ref) and
[`vector_math`](@ref) reach the operations without a Julia function.
"""
module TextureNodes

using ..Hikari: TexNode, NodeCore, NodeOp, MathOp, VectorMathOp, RangeType, BlendType, ColorModel, MappingType,
                RotateType, NoiseType, VoronoiFeature, VoronoiMetric, WaveType, WaveDirection, WaveProfile,
                GradientType, Texture, RGBSpectrum, to_texture, RAMP_TABLE_SIZE, RAMP_ENTRIES_PER_DATA, NODE_ARGS,
                _width, ShaderNode, BSDFNode, MixShaderNode, AddShaderNode, ClosureKind, shifted_graph
using LinearAlgebra: LinearAlgebra
using ..Hikari: Colorant, red, green, blue

export uv, xyz, normal, constant, combine, mix, smoothstep, fract, greater, less, transform, mapping, math,
       vector_math, map_range, color_ramp, hsv, gamma, bright_contrast, invert, rgb_to_bw, separate_color,
       combine_color, vector_rotate, noise, voronoi, white_noise, wave, gradient, magic, checker, brick,
       image_texture, rotation_matrix, principled_bsdf, diffuse_bsdf, glossy_bsdf, metallic_bsdf, translucent_bsdf,
       glass_bsdf, sheen_bsdf, transparent_bsdf, emission, mix_shader, add_shader, bump, normal_map, radial_tangent

const NodeLike = Union{TexNode, Real, Tuple{Real, Real, Real}, Colorant, AbstractVector{<:Real}}

constant(x::Real) = constant((x, x, x))
constant(v::Union{Tuple{Real, Real, Real}, AbstractVector{<:Real}}) =
    TexNode(NodeCore(NodeOp.CONST, (0x00, 0x00, 0x00), Any[Float32(v[1]), Float32(v[2]), Float32(v[3])], [3], nothing), 1)
constant(c::Colorant) = constant((red(c), green(c), blue(c)))

"""An input: nodes and plain numbers stay as they are, colours and vectors become constant nodes."""
input(x::TexNode) = x
input(x::Real) = Float32(x)
input(x) = constant(x)

width(x::TexNode) = _width(x)
width(::Real) = 1
width(x) = 3
widest(xs...) = maximum(width, xs)

node(op, inputs, widths; mode = (0x00, 0x00, 0x00), data = nothing) =
    TexNode(NodeCore(op, UInt8.(mode), Any[input(x) for x in inputs], collect(widths), data), 1)

"""Look a setting up by name, with the allowed names in the error."""
function setting(table, name::Symbol, what)
    haskey(table, name) || error("$what must be one of $(join(sort(collect(keys(table))), ", ")), got :$name")
    return table[name]
end

"""Texture coordinates `(u, v, 0)`."""
uv() = node(NodeOp.UV, (), (3,))
"""World-space position of the shading point."""
xyz() = node(NodeOp.POSITION, (), (3,))
"""Shading normal (world space)."""
normal() = node(NodeOp.NORMAL, (), (3,))

# ── Math ────────────────────────────────────────────────────────────────────

const MATH_OPS = Dict(
    :add => MathOp.ADD, :subtract => MathOp.SUBTRACT, :multiply => MathOp.MULTIPLY, :divide => MathOp.DIVIDE,
    :multiply_add => MathOp.MULTIPLY_ADD, :power => MathOp.POWER, :logarithm => MathOp.LOGARITHM,
    :sqrt => MathOp.SQRT, :inverse_sqrt => MathOp.INVERSE_SQRT, :absolute => MathOp.ABSOLUTE,
    :exponent => MathOp.EXPONENT, :minimum => MathOp.MINIMUM, :maximum => MathOp.MAXIMUM,
    :less_than => MathOp.LESS_THAN, :greater_than => MathOp.GREATER_THAN, :sign => MathOp.SIGN,
    :compare => MathOp.COMPARE, :smooth_min => MathOp.SMOOTH_MIN, :smooth_max => MathOp.SMOOTH_MAX,
    :round => MathOp.ROUND, :floor => MathOp.FLOOR, :ceil => MathOp.CEIL, :truncate => MathOp.TRUNCATE,
    :fraction => MathOp.FRACTION, :modulo => MathOp.MODULO, :floored_modulo => MathOp.FLOORED_MODULO,
    :wrap => MathOp.WRAP, :snap => MathOp.SNAP, :pingpong => MathOp.PINGPONG, :sine => MathOp.SINE,
    :cosine => MathOp.COSINE, :tangent => MathOp.TANGENT, :arcsine => MathOp.ARCSINE,
    :arccosine => MathOp.ARCCOSINE, :arctangent => MathOp.ARCTANGENT, :arctan2 => MathOp.ARCTAN2,
    :sinh => MathOp.SINH, :cosh => MathOp.COSH, :tanh => MathOp.TANH, :radians => MathOp.RADIANS,
    :degrees => MathOp.DEGREES)

"""
    math(op, a, b = 0.5, c = 0.5)

Blender's Math node, per component for vectors: `op` is one of its operations
by name, e.g. `:smooth_min`, `:wrap`, `:snap`, `:pingpong`, `:compare`.
"""
math(op::Symbol, a::NodeLike, b::NodeLike = 0.5, c::NodeLike = 0.5) =
    node(NodeOp.MATH, (a, b, c), (widest(a, b, c),); mode = (setting(MATH_OPS, op, "math operation"), 0, 0))

for (f, name) in ((:+, :add), (:-, :subtract), (:*, :multiply), (:/, :divide), (:^, :power), (:min, :minimum),
                  (:max, :maximum), (:mod, :floored_modulo), (:rem, :modulo), (:atan, :arctan2))
    @eval begin
        Base.$f(a::TexNode, b::TexNode) = math($(QuoteNode(name)), a, b)
        Base.$f(a::TexNode, b::NodeLike) = math($(QuoteNode(name)), a, b)
        Base.$f(a::NodeLike, b::TexNode) = math($(QuoteNode(name)), a, b)
    end
end
Base.:^(a::TexNode, b::Integer) = math(:power, a, b)
Base.log(b::Real, a::TexNode) = math(:logarithm, a, b)
Base.log(a::TexNode) = math(:logarithm, a, ℯ)

for (f, name) in ((:abs, :absolute), (:floor, :floor), (:ceil, :ceil), (:round, :round), (:trunc, :truncate),
                  (:sign, :sign), (:sin, :sine), (:cos, :cosine), (:tan, :tangent), (:asin, :arcsine),
                  (:acos, :arccosine), (:atan, :arctangent), (:sinh, :sinh), (:cosh, :cosh), (:tanh, :tanh),
                  (:sqrt, :sqrt), (:exp, :exponent), (:deg2rad, :radians), (:rad2deg, :degrees))
    @eval Base.$f(a::TexNode) = math($(QuoteNode(name)), a)
end
Base.:-(a::TexNode) = 0 - a

"""`x - floor(x)`."""
fract(x::NodeLike) = math(:fraction, x)
"""1 where `a > b`, else 0, per component."""
greater(a::NodeLike, b::NodeLike) = math(:greater_than, a, b)
"""1 where `a < b`, else 0, per component."""
less(a::NodeLike, b::NodeLike) = math(:less_than, a, b)

const VECTOR_OPS = Dict(
    :cross => VectorMathOp.CROSS, :project => VectorMathOp.PROJECT, :reflect => VectorMathOp.REFLECT,
    :refract => VectorMathOp.REFRACT, :faceforward => VectorMathOp.FACEFORWARD, :dot => VectorMathOp.DOT,
    :distance => VectorMathOp.DISTANCE, :length => VectorMathOp.LENGTH, :scale => VectorMathOp.SCALE,
    :normalize => VectorMathOp.NORMALIZE)
const SCALAR_VECTOR_OPS = (:dot, :distance, :length)

"""
    vector_math(op, a, b = 0, c = 0; scale = 1)

The operations of Blender's Vector Math node that are not per component:
`:cross`, `:project`, `:reflect`, `:refract` (with `scale` as the IOR ratio),
`:faceforward`, `:dot`, `:distance`, `:length`, `:scale`, `:normalize`.
"""
vector_math(op::Symbol, a::NodeLike, b::NodeLike = 0, c::NodeLike = 0; scale::NodeLike = 1) =
    node(NodeOp.VECTOR_MATH, (a, b, c, scale), (op in SCALAR_VECTOR_OPS ? 1 : 3,);
         mode = (setting(VECTOR_OPS, op, "vector math operation"), 0, 0))

LinearAlgebra.dot(a::TexNode, b::NodeLike) = vector_math(:dot, a, b)
LinearAlgebra.dot(a::NodeLike, b::TexNode) = vector_math(:dot, a, b)
LinearAlgebra.dot(a::TexNode, b::TexNode) = vector_math(:dot, a, b)
LinearAlgebra.cross(a::TexNode, b::NodeLike) = vector_math(:cross, a, b)
LinearAlgebra.cross(a::NodeLike, b::TexNode) = vector_math(:cross, a, b)
LinearAlgebra.cross(a::TexNode, b::TexNode) = vector_math(:cross, a, b)
LinearAlgebra.norm(a::TexNode) = vector_math(:length, a)
LinearAlgebra.normalize(a::TexNode) = vector_math(:normalize, a)

"""A vector from the first components of three scalars."""
combine(x::NodeLike, y::NodeLike, z::NodeLike) = node(NodeOp.COMBINE, (x, y, z), (3,))

"""
    clamp(x, lo, hi; range = false)

Blender's Clamp node; with `range = true`, `lo > hi` clamps to `[hi, lo]`.
"""
Base.clamp(x::TexNode, lo::NodeLike, hi::NodeLike; range::Bool = false) =
    node(NodeOp.CLAMP, (x, lo, hi), (widest(x, lo, hi),); mode = (range, 0, 0))

const RANGE_TYPES = Dict(:linear => RangeType.LINEAR, :stepped => RangeType.STEPPED,
                         :smoothstep => RangeType.SMOOTHSTEP, :smootherstep => RangeType.SMOOTHERSTEP)

"""
    map_range(x, from_min = 0, from_max = 1, to_min = 0, to_max = 1; interpolation = :linear, steps = 4, clamp = true)

Blender's Map Range node, for scalars or per component for vectors.
`interpolation` is `:linear`, `:stepped`, `:smoothstep` or `:smootherstep`.
"""
map_range(x::NodeLike, from_min::NodeLike = 0, from_max::NodeLike = 1, to_min::NodeLike = 0, to_max::NodeLike = 1;
          interpolation::Symbol = :linear, steps::NodeLike = 4, clamp::Bool = true) =
    node(NodeOp.MAP_RANGE, (x, from_min, from_max, to_min, to_max, steps),
         (widest(x, from_min, from_max, to_min, to_max, steps),);
         mode = (setting(RANGE_TYPES, interpolation, "map_range interpolation"), clamp, 0))

"""Hermite step from 0 at `edge0` to 1 at `edge1`."""
smoothstep(edge0::NodeLike, edge1::NodeLike, x::NodeLike) = map_range(x, edge0, edge1; interpolation = :smoothstep)

# ── Colour ──────────────────────────────────────────────────────────────────

const BLEND_TYPES = Dict(
    :mix => BlendType.MIX, :add => BlendType.ADD, :multiply => BlendType.MULTIPLY, :screen => BlendType.SCREEN,
    :overlay => BlendType.OVERLAY, :subtract => BlendType.SUBTRACT, :divide => BlendType.DIVIDE,
    :difference => BlendType.DIFFERENCE, :exclusion => BlendType.EXCLUSION, :darken => BlendType.DARKEN,
    :lighten => BlendType.LIGHTEN, :dodge => BlendType.DODGE, :burn => BlendType.BURN, :hue => BlendType.HUE,
    :saturation => BlendType.SATURATION, :value => BlendType.VALUE, :color => BlendType.COLOR,
    :soft_light => BlendType.SOFT_LIGHT, :linear_light => BlendType.LINEAR_LIGHT)

"""
    mix(a, b, factor; blend = :mix, clamp_factor = false, clamp_result = false)

Blender's Mix node: `a` towards `b` by `factor`, which may be a vector for the
plain `:mix`. `blend` is one of its colour blend modes, e.g. `:multiply`,
`:screen`, `:overlay`, `:soft_light`, `:hue`.
"""
mix(a::NodeLike, b::NodeLike, factor::NodeLike; blend::Symbol = :mix, clamp_factor::Bool = false,
    clamp_result::Bool = false) =
    node(NodeOp.MIX, (factor, a, b), (blend == :mix ? widest(a, b, factor) : 3,);
         mode = (setting(BLEND_TYPES, blend, "mix blend"), clamp_factor, clamp_result))

"""Blender's Hue/Saturation/Value node."""
hsv(color::NodeLike; hue::NodeLike = 0.5, saturation::NodeLike = 1, value::NodeLike = 1, fac::NodeLike = 1) =
    node(NodeOp.HSV, (hue, saturation, value, fac, color), (3,))

"""Blender's Gamma node."""
gamma(color::NodeLike, g::NodeLike = 1) = node(NodeOp.GAMMA, (color, g), (3,))

"""Blender's Brightness/Contrast node."""
bright_contrast(color::NodeLike; bright::NodeLike = 0, contrast::NodeLike = 0) =
    node(NodeOp.BRIGHT_CONTRAST, (color, bright, contrast), (3,))

"""Blender's Invert Color node."""
invert(color::NodeLike; fac::NodeLike = 1) = node(NodeOp.INVERT, (fac, color), (3,))

"""Luminance of a linear sRGB colour, as Blender's RGB to BW node."""
rgb_to_bw(color::NodeLike) = node(NodeOp.RGB_TO_BW, (color,), (1,))

const COLOR_MODELS = Dict(:rgb => ColorModel.RGB, :hsv => ColorModel.HSV, :hsl => ColorModel.HSL)

"""Blender's Separate Color node: a vector of the channels of `model` (`:rgb`, `:hsv` or `:hsl`), e.g. `.x` for hue."""
separate_color(color::NodeLike; model::Symbol = :hsv) =
    node(NodeOp.SEPARATE_COLOR, (color,), (3,); mode = (setting(COLOR_MODELS, model, "color model"), 0, 0))

"""Blender's Combine Color node: an RGB colour from three channels of `model`."""
combine_color(a::NodeLike, b::NodeLike, c::NodeLike; model::Symbol = :hsv) =
    node(NodeOp.COMBINE_COLOR, (combine(a, b, c),), (3,); mode = (setting(COLOR_MODELS, model, "color model"), 0, 0))

"""The value of a ramp at `t` between its stops, by `interpolation`."""
function ramp_value(stops, t, interpolation)
    positions = first.(stops)
    values = [Float64.(collect(last(s))) for s in stops]
    t <= positions[1] && return values[1]
    t >= positions[end] && return values[end]
    j = findlast(<=(t), positions)
    interpolation == :constant && return values[j]
    u = (t - positions[j]) / max(positions[j + 1] - positions[j], 1e-12)
    interpolation == :linear && return (1 - u) * values[j] + u * values[j + 1]
    interpolation == :ease && return values[j] + (3u^2 - 2u^3) * (values[j + 1] - values[j])
    p0, p1, p2, p3 = values[max(j - 1, 1)], values[j], values[j + 1], values[min(j + 2, end)]
    if interpolation == :cardinal
        return 0.5 * (2p1 + (p2 - p0) * u + (2p0 - 5p1 + 4p2 - p3) * u^2 + (3p1 - p0 - 3p2 + p3) * u^3)
    end
    b = ((1 - u)^3, 3u^3 - 6u^2 + 4, -3u^3 + 3u^2 + 3u + 1, u^3) ./ 6
    return b[1] * p0 + b[2] * p1 + b[3] * p2 + b[4] * p3
end

const RAMP_INTERPOLATIONS = (:constant, :linear, :ease, :cardinal, :b_spline)

"""The ramp's table, padded so every `DATA` instruction holds whole entries."""
function ramp_data(stops, interpolation)
    table = [Float32.(ramp_value(stops, i / (RAMP_TABLE_SIZE - 1), interpolation)) for i in 0:RAMP_TABLE_SIZE - 1]
    groups = Iterators.partition(table, RAMP_ENTRIES_PER_DATA)
    padding = zeros(Float32, NODE_ARGS - 3 * RAMP_ENTRIES_PER_DATA)
    return reduce(vcat, [vcat(reduce(vcat, g), zeros(Float32, 3 * (RAMP_ENTRIES_PER_DATA - length(g))), padding)
                         for g in groups])
end

"""
    color_ramp(t, stops...; interpolation = :linear)

Blender's Color Ramp: `stops` are `position => colour` pairs (or numbers) in
increasing position; `t` below the first or above the last stop holds the end
colour. `interpolation` is `:constant`, `:linear`, `:ease`, `:cardinal` or
`:b_spline`. Like Cycles, the ramp is baked into a table of $RAMP_TABLE_SIZE colours.
"""
function color_ramp(t::NodeLike, stops::Pair...; interpolation::Symbol = :linear)
    isempty(stops) && error("color_ramp needs at least one stop")
    issorted(first.(stops)) || error("color_ramp stops must be in increasing position")
    interpolation in RAMP_INTERPOLATIONS ||
        error("color_ramp interpolation must be one of $(join(RAMP_INTERPOLATIONS, ", ")), got :$interpolation")
    colours = [first(p) => (last(p) isa Real ? (last(p), last(p), last(p)) :
                     last(p) isa Colorant ? (red(last(p)), green(last(p)), blue(last(p))) : Tuple(last(p)))
               for p in stops]
    return node(NodeOp.RAMP, (t,), (3,); mode = (interpolation != :constant, 0, 0),
                data = ramp_data([Float64(first(p)) => last(p) for p in colours], interpolation))
end

# ── Vector ──────────────────────────────────────────────────────────────────

"""
    transform(p, M)

Apply an affine map to a point: `M` is a 4×4 matrix (only its top three rows
are used) or a 3×3 one. Use the inverse of an object's model matrix to get
coordinates that stay attached to the object.
"""
function transform(p::NodeLike, M::AbstractMatrix)
    size(M) in ((3, 3), (4, 4)) || error("transform needs a 3×3 or 4×4 matrix, got $(size(M))")
    t(i) = size(M, 2) == 4 ? M[i, 4] : 0
    row(i) = node(NodeOp.AFFINE_ROW, (p, 0, 0, 0, 0, M[i, 1], M[i, 2], M[i, 3], t(i)), (1,))
    return combine(row(1), row(2), row(3))
end

"""Rotation by Euler angles in radians, applied about x, then y, then z."""
function rotation_matrix((a, b, c))
    Rx = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)]
    Ry = [cos(b) 0 sin(b); 0 1 0; -sin(b) 0 cos(b)]
    Rz = [cos(c) -sin(c) 0; sin(c) cos(c) 0; 0 0 1]
    return Rz * Ry * Rx
end

const MAPPING_TYPES = Dict(:point => MappingType.POINT, :texture => MappingType.TEXTURE,
                           :vector => MappingType.VECTOR, :normal => MappingType.NORMAL)

"""
    mapping(p; location = (0, 0, 0), rotation = (0, 0, 0), scale = (1, 1, 1), type = :point)

Blender's Mapping node. For `:point`, scale, then rotate (Euler xyz, radians),
then translate; `:texture` is its inverse; `:vector` and `:normal` ignore the
location.
"""
mapping(p::NodeLike; location::NodeLike = (0, 0, 0), rotation::NodeLike = (0, 0, 0), scale::NodeLike = (1, 1, 1),
        type::Symbol = :point) =
    node(NodeOp.MAPPING, (p, location, rotation, scale), (3,); mode = (setting(MAPPING_TYPES, type, "mapping type"), 0, 0))

const ROTATE_TYPES = Dict(:axis_angle => RotateType.AXIS_ANGLE, :x => RotateType.X, :y => RotateType.Y,
                          :z => RotateType.Z, :euler_xyz => RotateType.EULER_XYZ)

"""
    vector_rotate(v; type = :axis_angle, center = (0, 0, 0), axis = (0, 0, 1), angle = 0, rotation = (0, 0, 0), invert = false)

Blender's Vector Rotate node: about `axis` (or the x, y or z axis) by `angle`,
or by the Euler angles `rotation` for `:euler_xyz`, around `center`.
"""
vector_rotate(v::NodeLike; type::Symbol = :axis_angle, center::NodeLike = (0, 0, 0), axis::NodeLike = (0, 0, 1),
              angle::NodeLike = 0, rotation::NodeLike = (0, 0, 0), invert::Bool = false) =
    node(NodeOp.VECTOR_ROTATE, (v, center, axis, angle, rotation), (3,);
         mode = (setting(ROTATE_TYPES, type, "vector_rotate type"), invert, 0))

# ── Textures ────────────────────────────────────────────────────────────────

dimensions(d::Integer) = d in 1:4 ? d : error("dimensions must be 1, 2, 3 or 4, got $d")

const NOISE_TYPES = Dict(:fbm => NoiseType.FBM, :multifractal => NoiseType.MULTIFRACTAL,
                         :hybrid_multifractal => NoiseType.HYBRID_MULTIFRACTAL,
                         :ridged_multifractal => NoiseType.RIDGED_MULTIFRACTAL,
                         :hetero_terrain => NoiseType.HETERO_TERRAIN)

"""
    noise(p = xyz(); scale = 5, detail = 2, roughness = 0.5, lacunarity = 2, distortion = 0,
          type = :fbm, normalize = true, offset = 0, gain = 1, w = 0, dimensions = 3)

Blender's Noise Texture. Its outputs are `fac` and `.color`. `type` is `:fbm`,
`:multifractal`, `:hybrid_multifractal`, `:ridged_multifractal` or
`:hetero_terrain`; `w` is the fourth coordinate for `dimensions = 4` and the
only one for `dimensions = 1`.
"""
noise(p::NodeLike = xyz(); scale::NodeLike = 5, detail::NodeLike = 2, roughness::NodeLike = 0.5,
      lacunarity::NodeLike = 2, distortion::NodeLike = 0, type::Symbol = :fbm, normalize::Bool = true,
      offset::NodeLike = 0, gain::NodeLike = 1, w::NodeLike = 0, dimensions::Integer = 3) =
    node(NodeOp.NOISE, (p, w, scale, detail, roughness, lacunarity, offset, gain, distortion), (1, 3);
         mode = (TextureNodes.dimensions(dimensions), setting(NOISE_TYPES, type, "noise type"), normalize))

const VORONOI_FEATURES = Dict(:f1 => VoronoiFeature.F1, :f2 => VoronoiFeature.F2,
                              :smooth_f1 => VoronoiFeature.SMOOTH_F1,
                              :distance_to_edge => VoronoiFeature.DISTANCE_TO_EDGE,
                              :n_sphere_radius => VoronoiFeature.N_SPHERE_RADIUS)
const VORONOI_METRICS = Dict(:euclidean => VoronoiMetric.EUCLIDEAN, :manhattan => VoronoiMetric.MANHATTAN,
                             :chebychev => VoronoiMetric.CHEBYCHEV, :minkowski => VoronoiMetric.MINKOWSKI)

"""
    voronoi(p = xyz(); scale = 5, detail = 0, roughness = 0.5, lacunarity = 2, smoothness = 1, exponent = 0.5,
            randomness = 1, feature = :f1, metric = :euclidean, normalize = false, w = 0, dimensions = 3)

Blender's Voronoi Texture. Its outputs are `distance`, `.color`, `.position`,
`.w` and `.radius` (for `feature = :n_sphere_radius`). `feature` is `:f1`,
`:f2`, `:smooth_f1`, `:distance_to_edge` or `:n_sphere_radius`; `metric` is
`:euclidean`, `:manhattan`, `:chebychev` or `:minkowski` (with `exponent`).
"""
function voronoi(p::NodeLike = xyz(); scale::NodeLike = 5, detail::NodeLike = 0, roughness::NodeLike = 0.5,
                 lacunarity::NodeLike = 2, smoothness::NodeLike = 1, exponent::NodeLike = 0.5,
                 randomness::NodeLike = 1, feature::Symbol = :f1, metric::Symbol = :euclidean,
                 normalize::Bool = false, w::NodeLike = 0, dimensions::Integer = 3)
    packed = setting(VORONOI_METRICS, metric, "voronoi metric") | (UInt8(normalize) << 4)
    return node(NodeOp.VORONOI, (p, w, scale, detail, roughness, lacunarity, smoothness, exponent, randomness),
                (1, 3, 3, 1, 1);
                mode = (TextureNodes.dimensions(dimensions), setting(VORONOI_FEATURES, feature, "voronoi feature"), packed))
end

"""
    white_noise(p = xyz(); w = 0, dimensions = 3)

Blender's White Noise Texture: a random `value` in [0, 1] and `.color` for
every distinct input, e.g. `white_noise(floor(p))` per cell.
"""
white_noise(p::NodeLike = xyz(); w::NodeLike = 0, dimensions::Integer = 3) =
    node(NodeOp.WHITE_NOISE, (p, w), (1, 3); mode = (TextureNodes.dimensions(dimensions), 0, 0))

const WAVE_TYPES = Dict(:bands => WaveType.BANDS, :rings => WaveType.RINGS)
const WAVE_DIRECTIONS = Dict(:x => WaveDirection.X, :y => WaveDirection.Y, :z => WaveDirection.Z,
                             :diagonal => WaveDirection.DIAGONAL, :spherical => WaveDirection.SPHERICAL)
const WAVE_PROFILES = Dict(:sine => WaveProfile.SINE, :saw => WaveProfile.SAW, :triangle => WaveProfile.TRIANGLE)

"""
    wave(p = xyz(); type = :bands, direction = :x, profile = :sine, scale = 5,
         distortion = 0, detail = 2, detail_scale = 1, detail_roughness = 0.5, phase = 0)

Blender's Wave Texture; its outputs are `color` and `.fac`. Bands run along
`:x`, `:y`, `:z` or `:diagonal`, rings around `:x`, `:y`, `:z` or `:spherical`.
"""
function wave(p::NodeLike = xyz(); type::Symbol = :bands, direction::Symbol = :x, profile::Symbol = :sine,
              scale::NodeLike = 5, distortion::NodeLike = 0, detail::NodeLike = 2, detail_scale::NodeLike = 1,
              detail_roughness::NodeLike = 0.5, phase::NodeLike = 0)
    t = setting(WAVE_TYPES, type, "wave type")
    allowed = type == :bands ? (:x, :y, :z, :diagonal) : (:x, :y, :z, :spherical)
    direction in allowed || error("$type wave direction must be one of $(join(allowed, ", ")), got :$direction")
    return node(NodeOp.WAVE, (p, scale, distortion, detail, detail_scale, detail_roughness, phase), (3, 1);
                mode = (t, WAVE_DIRECTIONS[direction], setting(WAVE_PROFILES, profile, "wave profile")))
end

const GRADIENT_TYPES = Dict(:linear => GradientType.LINEAR, :quadratic => GradientType.QUADRATIC,
                            :easing => GradientType.EASING, :diagonal => GradientType.DIAGONAL,
                            :radial => GradientType.RADIAL, :quadratic_sphere => GradientType.QUADRATIC_SPHERE,
                            :spherical => GradientType.SPHERICAL)

"""Blender's Gradient Texture; its outputs are `color` and `.fac`."""
gradient(p::NodeLike = xyz(); type::Symbol = :linear) =
    node(NodeOp.GRADIENT, (p,), (3, 1); mode = (setting(GRADIENT_TYPES, type, "gradient type"), 0, 0))

"""Blender's Magic Texture; its outputs are `color` and `.fac`."""
function magic(p::NodeLike = xyz(); depth::Integer = 2, scale::NodeLike = 5, distortion::NodeLike = 1)
    0 <= depth <= 10 || error("magic depth must be between 0 and 10, got $depth")
    return node(NodeOp.MAGIC, (p, scale, distortion), (3, 1); mode = (depth, 0, 0))
end

"""Blender's Checker Texture; its outputs are `color` and `.fac`."""
checker(p::NodeLike = xyz(); color1::NodeLike = (0.8, 0.8, 0.8), color2::NodeLike = (0.2, 0.2, 0.2),
        scale::NodeLike = 5) =
    node(NodeOp.CHECKER, (p, color1, color2, scale), (3, 1))

"""
    brick(p = xyz(); color1, color2, mortar, scale = 5, mortar_size = 0.02, mortar_smooth = 0.1, bias = 0,
          brick_width = 0.5, row_height = 0.25, offset = 0.5, offset_frequency = 2, squash = 1, squash_frequency = 2)

Blender's Brick Texture; its outputs are `color` and `.fac` (1 in the mortar).
"""
function brick(p::NodeLike = xyz(); color1::NodeLike = (0.8, 0.8, 0.8), color2::NodeLike = (0.2, 0.2, 0.2),
               mortar::NodeLike = (0, 0, 0), scale::NodeLike = 5, mortar_size::NodeLike = 0.02,
               mortar_smooth::NodeLike = 0.1, bias::NodeLike = 0, brick_width::NodeLike = 0.5,
               row_height::NodeLike = 0.25, offset::Real = 0.5, offset_frequency::Integer = 2, squash::Real = 1,
               squash_frequency::Integer = 2)
    1 <= offset_frequency <= 99 || error("brick offset_frequency must be between 1 and 99, got $offset_frequency")
    1 <= squash_frequency <= 99 || error("brick squash_frequency must be between 1 and 99, got $squash_frequency")
    return node(NodeOp.BRICK, (p, color1, color2, mortar, scale, mortar_size, mortar_smooth, bias, brick_width,
                               row_height), (3, 1);
                mode = (offset_frequency, squash_frequency, 0), data = Float32[offset, squash])
end

"""
    image_texture(img, coords = uv())

Sample an image (a matrix of colours or numbers, or a `Hikari.Texture`) at the
first two components of `coords`, wrapping like a uv lookup.
"""
function image_texture(img, coords::NodeLike = uv())
    data = img isa Texture ? img.data : img isa AbstractMatrix{<:Colorant} ? to_texture(img).data : img
    ndims(data) == 2 || error("image needs a 2D image, got $(ndims(data)) dimensions")
    return node(NodeOp.IMAGE, (coords,), (3,); data)
end


# ── Shaders ─────────────────────────────────────────────────────────────────

"""The tangent Blender gives an anisotropic BSDF without one: around the Z axis, `(-y, x, 0)` of the position."""
radial_tangent() = (p = xyz(); combine(-p.y, p.x, 0))

"""
    principled_bsdf(; base_color = (0.8, 0.8, 0.8), metallic = 0, roughness = 0.5, ior = 1.5, alpha = 1,
                    normal = normal(), diffuse_roughness = 0, subsurface_weight = 0, subsurface_radius = (1, 0.2, 0.1),
                    subsurface_scale = 0.05, subsurface_ior = 1.4, subsurface_anisotropy = 0, specular_ior_level = 0.5,
                    specular_tint = (1, 1, 1), anisotropic = 0, anisotropic_rotation = 0, tangent = radial_tangent(),
                    transmission_weight = 0, transmission_dispersion_scale = 0, transmission_dispersion_abbe_number = 20,
                    coat_weight = 0, coat_roughness = 0.03, coat_ior = 1.5, coat_tint = (1, 1, 1), coat_normal = normal,
                    sheen_weight = 0, sheen_roughness = 0.5, sheen_tint = (1, 1, 1), emission_color = (1, 1, 1),
                    emission_strength = 0, thin_film_thickness = 0, thin_film_ior = 1.33, thin_wall = false)

Blender's Principled BSDF with every input it has, laid out as Cycles does: transparency, sheen
and coat over the metal, transmission and the specular layer over the diffuse base. With
`thin_wall`, transmission and subsurface are those of a surface with no thickness. The default `tangent`
is Blender's, radial around the Z axis (here the world's, Blender's is the object's). Subsurface scattering is approximated by a diffuse lobe: the
radius, scale, IOR and anisotropy inputs are accepted but do not change it yet.
"""
function principled_bsdf(; base_color::NodeLike = (0.8, 0.8, 0.8), metallic::NodeLike = 0, roughness::NodeLike = 0.5,
                         ior::NodeLike = 1.5, alpha::NodeLike = 1, normal::NodeLike = TextureNodes.normal(),
                         diffuse_roughness::NodeLike = 0, subsurface_weight::NodeLike = 0,
                         subsurface_radius::NodeLike = (1, 0.2, 0.1), subsurface_scale::NodeLike = 0.05,
                         subsurface_ior::NodeLike = 1.4, subsurface_anisotropy::NodeLike = 0,
                         specular_ior_level::NodeLike = 0.5, specular_tint::NodeLike = (1, 1, 1),
                         anisotropic::NodeLike = 0, anisotropic_rotation::NodeLike = 0,
                         tangent::NodeLike = radial_tangent(),
                         transmission_weight::NodeLike = 0, transmission_dispersion_scale::NodeLike = 0,
                         transmission_dispersion_abbe_number::NodeLike = 20, coat_weight::NodeLike = 0,
                         coat_roughness::NodeLike = 0.03, coat_ior::NodeLike = 1.5, coat_tint::NodeLike = (1, 1, 1),
                         coat_normal::NodeLike = normal, sheen_weight::NodeLike = 0, sheen_roughness::NodeLike = 0.5,
                         sheen_tint::NodeLike = (1, 1, 1), emission_color::NodeLike = (1, 1, 1),
                         emission_strength::NodeLike = 0, thin_film_thickness::NodeLike = 0,
                         thin_film_ior::NodeLike = 1.33, thin_wall::Union{Bool, NodeLike} = false)
    return bsdf(NodeOp.PRINCIPLED, 0x00, base_color, metallic, roughness, ior, alpha, normal, diffuse_roughness,
                subsurface_weight, subsurface_radius, subsurface_scale, subsurface_ior, subsurface_anisotropy,
                specular_ior_level, specular_tint, anisotropic, anisotropic_rotation, tangent, transmission_weight,
                transmission_dispersion_scale, transmission_dispersion_abbe_number, coat_weight, coat_roughness,
                coat_ior, coat_tint, coat_normal, sheen_weight, sheen_roughness, sheen_tint, emission_color,
                emission_strength, thin_film_thickness, thin_film_ior, thin_wall isa Bool ? Float32(thin_wall) : thin_wall)
end

"""    diffuse_bsdf(; color = (0.8, 0.8, 0.8), roughness = 0, normal = normal())

Blender's Diffuse BSDF: Lambertian, or Cycles' energy-preserving Oren-Nayar with `roughness`."""
diffuse_bsdf(; color::NodeLike = (0.8, 0.8, 0.8), roughness::NodeLike = 0, normal::NodeLike = TextureNodes.normal()) =
    bsdf(NodeOp.CLOSURE, ClosureKind.DIFFUSE, color, roughness, normal)

"""    glossy_bsdf(; color = (0.8, 0.8, 0.8), roughness = 0.5, anisotropy = 0, rotation = 0, normal = normal(), tangent = (0, 0, 0))

Blender's Glossy BSDF with the multiscatter GGX distribution: a microfacet reflection tinted by `color` at every angle."""
glossy_bsdf(; color::NodeLike = (0.8, 0.8, 0.8), roughness::NodeLike = 0.5, anisotropy::NodeLike = 0,
            rotation::NodeLike = 0, normal::NodeLike = TextureNodes.normal(), tangent::NodeLike = radial_tangent()) =
    bsdf(NodeOp.CLOSURE, ClosureKind.GLOSSY, color, roughness, anisotropy, rotation, normal, tangent)

"""    metallic_bsdf(; base_color = (0.617, 0.577, 0.540), edge_tint = (0.695, 0.726, 0.770), roughness = 0.5, anisotropy = 0, rotation = 0, normal = normal(), tangent = (0, 0, 0))

Blender's Metallic BSDF with the F82 Tint Fresnel."""
metallic_bsdf(; base_color::NodeLike = (0.617, 0.577, 0.540), edge_tint::NodeLike = (0.695, 0.726, 0.770),
              roughness::NodeLike = 0.5, anisotropy::NodeLike = 0, rotation::NodeLike = 0,
              normal::NodeLike = TextureNodes.normal(), tangent::NodeLike = radial_tangent()) =
    bsdf(NodeOp.CLOSURE, ClosureKind.METALLIC, base_color, edge_tint, roughness, anisotropy, rotation, normal, tangent)

"""    glass_bsdf(; color = (1, 1, 1), roughness = 0, ior = 1.5, normal = normal())

Blender's Glass BSDF: GGX reflection and refraction, both tinted by `color`."""
glass_bsdf(; color::NodeLike = (1, 1, 1), roughness::NodeLike = 0, ior::NodeLike = 1.5,
           normal::NodeLike = TextureNodes.normal()) =
    bsdf(NodeOp.CLOSURE, ClosureKind.GLASS, color, roughness, ior, normal)

"""    translucent_bsdf(; color = (0.8, 0.8, 0.8), normal = normal())

Blender's Translucent BSDF: diffuse light through the surface."""
translucent_bsdf(; color::NodeLike = (0.8, 0.8, 0.8), normal::NodeLike = TextureNodes.normal()) =
    bsdf(NodeOp.CLOSURE, ClosureKind.TRANSLUCENT, color, normal)

"""    sheen_bsdf(; color = (0.8, 0.8, 0.8), roughness = 0.5, normal = normal())

Blender's Sheen BSDF with the microfiber distribution."""
sheen_bsdf(; color::NodeLike = (0.8, 0.8, 0.8), roughness::NodeLike = 0.5, normal::NodeLike = TextureNodes.normal()) =
    bsdf(NodeOp.CLOSURE, ClosureKind.SHEEN, color, roughness, normal)

"""    transparent_bsdf(; color = (1, 1, 1))

Blender's Transparent BSDF: light passes through unrefracted, by `color` (averaged to a coverage)."""
transparent_bsdf(; color::NodeLike = (1, 1, 1)) = bsdf(NodeOp.CLOSURE, ClosureKind.TRANSPARENT, color)

"""    emission(; color = (1, 1, 1), strength = 1)

Blender's Emission: radiance `strength` times `color`, found by the paths that hit it (it is not a light for direct lighting)."""
emission(; color::NodeLike = (1, 1, 1), strength::NodeLike = 1) = bsdf(NodeOp.CLOSURE, ClosureKind.EMISSION, color, strength)

"""    mix_shader(fac, a, b)

Blender's Mix Shader: `a` where `fac` is 0, `b` where it is 1."""
mix_shader(fac::NodeLike, a::ShaderNode, b::ShaderNode) = MixShaderNode(input(fac), a, b)

"""    add_shader(a, b)

Blender's Add Shader: the sum of both, which is not energy conserving in general."""
add_shader(a::ShaderNode, b::ShaderNode) = AddShaderNode(a, b)

bsdf(op, kind, inputs...) = BSDFNode(op, (kind, 0x00, 0x00), Any[input(x) for x in inputs])

"""
    bump(height; strength = 1, distance = 0.001, filter_width = 0.1, invert = false, normal = normal())

Blender's Bump node: the normal of the surface displaced by `distance` times `height` along it,
for a shader's `normal` input. As in Cycles, the height graph is evaluated at the hit and at the
hit moved by `filter_width` of a pixel in either screen direction.
"""
function bump(height::NodeLike; strength::NodeLike = 1, distance::NodeLike = 0.001, filter_width::Real = 0.1,
              invert::Bool = false, normal::NodeLike = TextureNodes.normal())
    h = input(height)
    fw = Float32(filter_width)
    h_dx, h_dy = h isa TexNode ? (shifted_graph(h, :dx, fw), shifted_graph(h, :dy, fw)) : (h, h)
    return node(NodeOp.BUMP, (h, h_dx, h_dy, strength, distance, normal, fw), (3,); mode = (invert, 0, 0))
end

const NORMAL_MAP_SPACES = Dict(:tangent => 0x00, :world => 0x02, :blender_world => 0x04)

"""
    normal_map(color; strength = 1, space = :tangent, invert_green = false)

Blender's Normal Map node: the normal a normal map's `color` encodes, for a shader's `normal`
input. In `:tangent` space the colour is read in the frame of the uv map's tangent, bitangent and
the shading normal (OpenGL convention: green along increasing v; `invert_green` reads DirectX
maps); `:world` and `:blender_world` take it as a world space normal. Object space is not
supported.
"""
function normal_map(color::NodeLike; strength::NodeLike = 1, space::Symbol = :tangent, invert_green::Bool = false)
    return node(NodeOp.NORMAL_MAP, (color, strength), (3,);
                mode = (setting(NORMAL_MAP_SPACES, space, "normal map space"), invert_green, 0))
end

end
