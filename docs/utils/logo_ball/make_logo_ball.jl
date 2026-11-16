# The Makie logo ball of the docs' material renders: a unit sphere with the logo engraved and
# three grooves, built as a signed distance field, meshed with marching cubes, decimated in
# Blender and published as the `logo_ball` artifact. To make it again:
#
#     BLENDER=/Applications/Blender.app/Contents/MacOS/Blender julia -t auto --project=docs/utils/logo_ball docs/utils/logo_ball/make_logo_ball.jl out_dir
#     include("assets/publish.jl"); publish_artifact("logo_ball", out_dir; lazy = true)

using Downloads, GeometryBasics, LinearAlgebra, Meshing, Printf

const LOGO_SVG_URL = "https://raw.githubusercontent.com/MakieOrg/Makie.jl/master/docs/src/public/logo.svg"
const LOGO_CENTER = (200.0, 211.0)
const LOGO_HALF_EXTENT = 200.0

function logo_path_data(svg = Downloads.download(LOGO_SVG_URL))
    clip = match(r"<clipPath id=\"clip-1\">\s*<path[^>]* d=\"([^\"]+)\"", read(svg, String))
    return clip.captures[1]
end

cubic(p0, p1, p2, p3, t) = (1 - t)^3 * p0 + 3(1 - t)^2 * t * p1 + 3(1 - t) * t^2 * p2 + t^3 * p3

function logo_loops(d; samples_per_curve = 24)
    tokens = [m.match for m in eachmatch(r"[MLCZ]|-?\d*\.?\d+", d)]
    number(i) = parse(Float64, tokens[i])
    loops, i = Vector{Vector{Vec2}}(), 1
    while i <= length(tokens)
        command = tokens[i]
        i += 1
        if command == "M"
            push!(loops, [Vec2(number(i), number(i + 1))])
            i += 2
        elseif command == "L"
            push!(loops[end], Vec2(number(i), number(i + 1)))
            i += 2
        elseif command == "C"
            p0 = loops[end][end]
            p1, p2, p3 = (Vec2(number(i + k), number(i + k + 1)) for k in (0, 2, 4))
            append!(loops[end], [cubic(p0, p1, p2, p3, t) for t in range(0, 1; length = samples_per_curve + 1)[2:end]])
            i += 6
        end
    end
    return filter(loop -> length(loop) > 2, loops)
end

function to_local(p, logo_radius)
    scale = logo_radius / LOGO_HALF_EXTENT
    return Vec2((p[1] - LOGO_CENTER[1]) * scale, -(p[2] - LOGO_CENTER[2]) * scale)
end

function segment_distance(p, a, b)
    ab = b - a
    t = clamp(dot(p - a, ab) / dot(ab, ab), 0, 1)
    return norm(p - (a + t * ab))
end

crosses(p, a, b) = (a[2] > p[2]) != (b[2] > p[2]) && p[1] < a[1] + (p[2] - a[2]) / (b[2] - a[2]) * (b[1] - a[1])

function signed_distance(p, segments)
    distance, inside = Inf, false
    for (a, b) in segments
        distance = min(distance, segment_distance(p, a, b))
        crosses(p, a, b) && (inside = !inside)
    end
    return inside ? -distance : distance
end

struct Grid2D
    values::Matrix{Float64}
    extent::Float64
end

function Grid2D(segments; extent = 1.05, n = 1401)
    coordinate(i) = -extent + 2extent * (i - 1) / (n - 1)
    values = zeros(n, n)
    Threads.@threads for j in 1:n
        for i in 1:n
            values[i, j] = signed_distance(Vec2(coordinate(i), coordinate(j)), segments)
        end
    end
    return Grid2D(values, extent)
end

function (grid::Grid2D)(x, y)
    n = size(grid.values, 1)
    u = clamp((x + grid.extent) / (2grid.extent) * (n - 1) + 1, 1, n - 1e-9)
    v = clamp((y + grid.extent) / (2grid.extent) * (n - 1) + 1, 1, n - 1e-9)
    i, j = floor(Int, u), floor(Int, v)
    s, t = u - i, v - j
    g = grid.values
    return (1 - s) * (1 - t) * g[i, j] + s * (1 - t) * g[i + 1, j] + (1 - s) * t * g[i, j + 1] + s * t * g[i + 1, j + 1]
end

function round_intersection(a, b, r)
    r == 0 && return max(a, b)
    return min(-r, max(a, b)) + norm(Vec2(max(r + a, 0), max(r + b, 0)))
end

"""
Distance to a groove of tube radius `radius` along the unit sphere's circle at height `height`,
leaving out the arc within `stop` of the `-y` axis, with rounded ends.
"""
function groove_distance(p, height, radius, stop)
    circle_radius = sqrt(1 - height^2)
    azimuth = atan(p[1], -p[2])
    if abs(azimuth) >= stop
        return norm(Vec2(norm(Vec2(p[1], p[2])) - circle_radius, p[3] - height)) - radius
    end
    side = sign(azimuth) == 0 ? 1 : sign(azimuth)
    groove_end = Vec3(side * circle_radius * sin(stop), -circle_radius * cos(stop), height)
    return norm(p - groove_end) - radius
end

function engraved_sdf(; rim_radius, floor_radius, edge_depth = 0.15, logo_radius = 0.8125,
                      grooves = (), groove_radius = 0.03)
    loops = [[to_local(p, logo_radius) for p in loop] for loop in logo_loops(logo_path_data())]
    segments = [(loop[k], loop[mod1(k + 1, length(loop))]) for loop in loops for k in eachindex(loop)]
    outermost = maximum(norm(p) for loop in loops for p in loop)
    floor_y = -sqrt(1 - outermost^2) + edge_depth
    logo = Grid2D(segments)
    function sdf(p)
        cutter = round_intersection(logo(p[1], p[3]), p[2] - floor_y, floor_radius)
        engraved = round_intersection(norm(p) - 1, -cutter, rim_radius)
        isempty(grooves) && return engraved
        groove = minimum(((height, stop),) -> groove_distance(p, height, groove_radius, stop), grooves)
        return round_intersection(engraved, -groove, rim_radius)
    end
    return sdf
end

function sdf_gradient(sdf, p; h = 1e-3)
    e = (Vec3(h, 0, 0), Vec3(0, h, 0), Vec3(0, 0, h))
    return normalize(Vec3(map(ek -> (sdf(p + ek) - sdf(p - ek)) / 2h, e)))
end

function engraved_mesh(; resolution = 320, kwargs...)
    sdf = engraved_sdf(; kwargs...)
    axis = range(-1.03, 1.03; length = resolution)
    values = Array{Float64}(undef, resolution, resolution, resolution)
    Threads.@threads for k in 1:resolution
        for j in 1:resolution, i in 1:resolution
            values[i, j, k] = sdf(Vec3(axis[i], axis[j], axis[k]))
        end
    end
    vertices, triangles = isosurface(values, MarchingCubes(), axis, axis, axis)
    positions = [Point3f(v...) for v in vertices]
    vertex_normals = [Vec3f(sdf_gradient(sdf, Vec3(v...))) for v in vertices]
    return GeometryBasics.Mesh(positions, [TriangleFace{Int}(t...) for t in triangles]; normal = vertex_normals)
end

rotation_z(angle) = Mat3f(cos(angle), sin(angle), 0, -sin(angle), cos(angle), 0, 0, 0, 1)
rotation_x(angle) = Mat3f(1, 0, 0, 0, cos(angle), sin(angle), 0, -sin(angle), cos(angle))
const logo_turn = rotation_z(deg2rad(20)) * rotation_x(deg2rad(-15))

function write_obj(path, positions, triangles)
    open(path, "w") do io
        foreach(p -> @printf(io, "v %.5f %.5f %.5f\n", p...), positions)
        foreach(t -> println(io, "f ", join(Int.(t), " ")), triangles)
    end
end

function read_obj(path)
    lines = split.(eachline(path))
    positions = [Point3f(parse.(Float32, l[2:4])) for l in lines if l[1] == "v"]
    triangles = [TriangleFace{Int}((parse(Int, first(split(c, "/"))) for c in l[2:4])...) for l in lines if l[1] == "f"]
    return positions, triangles
end

function make_logo_ball(out_dir; decimate_ratio = 0.025)
    mesh = engraved_mesh(; rim_radius = 0.0225, floor_radius = 0.03,
                         grooves = ((-0.09, deg2rad(48)), (0.0, deg2rad(45)), (0.09, deg2rad(48))))
    full, decimated = joinpath(tempdir(), "logo_ball_full.obj"), joinpath(tempdir(), "logo_ball_decimated.obj")
    write_obj(full, coordinates(mesh), faces(mesh))
    run(`$(get(ENV, "BLENDER", "blender")) --background --python $(joinpath(@__DIR__, "decimate.py")) -- $full $decimated $decimate_ratio`)
    positions, triangles = read_obj(decimated)
    write_obj(joinpath(mkpath(out_dir), "logo_ball.obj"), [logo_turn * p for p in positions], triangles)
end

abspath(PROGRAM_FILE) == @__FILE__() && make_logo_ball(ARGS[1])
