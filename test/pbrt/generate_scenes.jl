# ============================================================================
# Scene generator for pbrt reference tests
#
# Generates .pbrt files covering:
#   - All material types × light types
#   - Roughness variants per material
#   - Texture variants (checkerboard reflectance)
#   - Filter variants (box, gaussian, mitchell, lanczos, triangle)
#   - Sensor variants (different ISO, white balance)
#   - Media (NanoVDB Perlin/Worley cloud + homogeneous tinted in dielectric sphere)
#
# Settings: 128x128, regularize=true, maxcomponentvalue=10
# Camera: zoomed in on sphere (dist=1.2, fov=40)
# ============================================================================

const SCENES_DIR = joinpath(@__DIR__, "scenes")
mkpath(SCENES_DIR)

# ============================================================================
# Scene defaults
# ============================================================================

const RES = 128
const CAMERA = """LookAt 0 -1.2 0.6   0 0 0.5   0 0 1
Camera "perspective" "float fov" 40"""
const FILM_BASE = """Film "rgb" "integer xresolution" $RES "integer yresolution" $RES "float maxcomponentvalue" 10"""
const INTEGRATOR = """Integrator "volpath" "integer maxdepth" 8 "bool regularize" true"""

# ============================================================================
# UV Sphere mesh generation with normals and UVs
# ============================================================================

function generate_uv_sphere(center=(0.0, 0.0, 0.5), radius=0.5;
                            n_lat=32, n_lon=64)
    vertices = Tuple{Float64,Float64,Float64}[]
    normals = Tuple{Float64,Float64,Float64}[]
    uvs = Tuple{Float64,Float64}[]
    indices = Tuple{Int,Int,Int}[]

    for i in 0:n_lat
        theta = pi * i / n_lat
        v = i / n_lat
        for j in 0:n_lon
            phi = 2pi * j / n_lon
            u = j / n_lon
            nx = sin(theta) * cos(phi)
            ny = sin(theta) * sin(phi)
            nz = cos(theta)
            push!(vertices, (center[1] + radius * nx, center[2] + radius * ny, center[3] + radius * nz))
            push!(normals, (nx, ny, nz))
            push!(uvs, (u, v))
        end
    end

    for i in 0:n_lat-1, j in 0:n_lon-1
        row = i * (n_lon + 1)
        next_row = (i + 1) * (n_lon + 1)
        push!(indices, (row + j, next_row + j, next_row + j + 1))
        push!(indices, (row + j, next_row + j + 1, row + j + 1))
    end

    return vertices, normals, uvs, indices
end

function generate_ground_disk(radius=2.0, n_segments=64)
    vertices = Tuple{Float64,Float64,Float64}[]
    normals = Tuple{Float64,Float64,Float64}[]
    uvs = Tuple{Float64,Float64}[]
    indices = Tuple{Int,Int,Int}[]

    push!(vertices, (0.0, 0.0, 0.0))
    push!(normals, (0.0, 0.0, 1.0))
    push!(uvs, (0.5, 0.5))

    for i in 0:n_segments
        angle = 2pi * i / n_segments
        push!(vertices, (radius * cos(angle), radius * sin(angle), 0.0))
        push!(normals, (0.0, 0.0, 1.0))
        push!(uvs, (0.5 + 0.5 * cos(angle), 0.5 + 0.5 * sin(angle)))
    end

    for i in 0:n_segments-1
        push!(indices, (0, i + 1, i + 2))
    end

    return vertices, normals, uvs, indices
end

# ============================================================================
# PBRT formatting
# ============================================================================

fmt_p3(pts) = join(["$(round(p[1],digits=6)) $(round(p[2],digits=6)) $(round(p[3],digits=6))" for p in pts], "  ")
fmt_p2(pts) = join(["$(round(p[1],digits=6)) $(round(p[2],digits=6))" for p in pts], "  ")
fmt_idx(tris) = join(["$(t[1]) $(t[2]) $(t[3])" for t in tris], "  ")

function write_mesh(io, verts, norms, uvs_arr, inds; indent="")
    println(io, "$(indent)Shape \"trianglemesh\"")
    println(io, "$(indent)  \"point3 P\" [ $(fmt_p3(verts)) ]")
    println(io, "$(indent)  \"normal N\" [ $(fmt_p3(norms)) ]")
    println(io, "$(indent)  \"point2 uv\" [ $(fmt_p2(uvs_arr)) ]")
    println(io, "$(indent)  \"integer indices\" [ $(fmt_idx(inds)) ]")
end

# Pre-generate meshes (shared across all scenes)
const SPHERE = generate_uv_sphere((0.0, 0.0, 0.5), 0.5; n_lat=32, n_lon=64)
const GROUND = generate_ground_disk(2.0, 64)

# ============================================================================
# Light definitions
# ============================================================================

const LIGHTS = Dict(
    "point"   => """LightSource "point" "rgb I" [40 40 40] "point3 from" [2 -1.5 3]""",
    "distant" => """LightSource "distant" "rgb L" [3.0 3.0 3.0] "point3 from" [2 -1 3] "point3 to" [0 0 0.5]""",
    "spot"    => """LightSource "spot" "rgb I" [80 80 80] "point3 from" [1.5 -1 2.5] "point3 to" [0 0 0.5] "float coneangle" 30 "float conedeltaangle" 5""",
    "area"    => """AttributeBegin
AreaLightSource "diffuse" "rgb L" [8 8 8] "bool twosided" true
Material "diffuse" "rgb reflectance" [1 1 1]
Translate 0.8 -0.4 1.5
Shape "trianglemesh"
  "point3 P" [ -0.4 -0.4 0  0.4 -0.4 0  0.4 0.4 0  -0.4 0.4 0 ]
  "integer indices" [ 0 1 2  0 2 3 ]
AttributeEnd""",
    "ambient" => """LightSource "infinite" "rgb L" [0.5 0.5 0.5]""",
)

# Extended light variants for light parameter tests
const LIGHT_VARIANTS = Dict(
    # Point light variants
    "point_warm"       => """LightSource "point" "rgb I" [50 30 10] "point3 from" [2 -1.5 3]""",
    "point_bright"     => """LightSource "point" "rgb I" [200 200 200] "point3 from" [2 -1.5 3]""",
    "point_blue"       => """LightSource "point" "rgb I" [10 25 60] "point3 from" [2 -1.5 3]""",
    "point_side"       => """LightSource "point" "rgb I" [40 40 40] "point3 from" [-2 -1.5 2]""",
    "point_top"        => """LightSource "point" "rgb I" [40 40 40] "point3 from" [0 0 4]""",
    "point_close"      => """LightSource "point" "rgb I" [10 10 10] "point3 from" [0.6 -0.4 1.0]""",
    # Spot light variants
    "spot_narrow"      => """LightSource "spot" "rgb I" [120 120 120] "point3 from" [1.5 -1 2.5] "point3 to" [0 0 0.5] "float coneangle" 15 "float conedeltaangle" 2""",
    "spot_wide"        => """LightSource "spot" "rgb I" [40 40 40] "point3 from" [1.5 -1 2.5] "point3 to" [0 0 0.5] "float coneangle" 60 "float conedeltaangle" 10""",
    "spot_side"        => """LightSource "spot" "rgb I" [80 80 80] "point3 from" [-1.5 -1 1.5] "point3 to" [0 0 0.5] "float coneangle" 25 "float conedeltaangle" 5""",
    # Distant light variants
    "distant_colored"  => """LightSource "distant" "rgb L" [2.0 0.5 0.5] "point3 from" [2 -1 3] "point3 to" [0 0 0.5]""",
    "distant_blue"     => """LightSource "distant" "rgb L" [0.5 0.5 2.0] "point3 from" [2 -1 3] "point3 to" [0 0 0.5]""",
    "distant_low"      => """LightSource "distant" "rgb L" [3.0 3.0 3.0] "point3 from" [3 -0.5 0.5] "point3 to" [0 0 0.5]""",
    # Area light variants
    "area_colored"     => """AttributeBegin
AreaLightSource "diffuse" "rgb L" [12 4 2] "bool twosided" true
Material "diffuse" "rgb reflectance" [1 1 1]
Translate 0.8 -0.4 1.5
Shape "trianglemesh"
  "point3 P" [ -0.4 -0.4 0  0.4 -0.4 0  0.4 0.4 0  -0.4 0.4 0 ]
  "integer indices" [ 0 1 2  0 2 3 ]
AttributeEnd""",
    "area_large"       => """AttributeBegin
AreaLightSource "diffuse" "rgb L" [3 3 3] "bool twosided" true
Material "diffuse" "rgb reflectance" [1 1 1]
Translate 0.8 -0.4 1.5
Shape "trianglemesh"
  "point3 P" [ -1.0 -1.0 0  1.0 -1.0 0  1.0 1.0 0  -1.0 1.0 0 ]
  "integer indices" [ 0 1 2  0 2 3 ]
AttributeEnd""",
    # Multi-light
    "point_plus_ambient" => """LightSource "point" "rgb I" [30 30 30] "point3 from" [2 -1.5 3]
LightSource "infinite" "rgb L" [0.15 0.15 0.15]""",
    "two_points"         => """LightSource "point" "rgb I" [30 10 5] "point3 from" [2 -1.5 3]
LightSource "point" "rgb I" [5 10 30] "point3 from" [-2 -1.5 3]""",
)

# ============================================================================
# Material definitions
# ============================================================================

const MATERIALS = Dict(
    # Diffuse
    "diffuse"         => """Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]""",
    "diffuse_colored" => """Material "diffuse" "rgb reflectance" [0.8 0.2 0.1]""",

    # Conductor
    "conductor_gold"   => """Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "float roughness" 0.01""",
    "conductor_mirror" => """Material "conductor" "float roughness" 0""",
    "conductor_rough"  => """Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "float roughness" 0.2""",
    "conductor_silver" => """Material "conductor" "spectrum eta" "metal-Ag-eta" "spectrum k" "metal-Ag-k" "float roughness" 0.05""",
    "conductor_copper" => """Material "conductor" "spectrum eta" "metal-Cu-eta" "spectrum k" "metal-Cu-k" "float roughness" 0.1""",
    "conductor_reflectance" => """Material "conductor" "rgb reflectance" [0.9 0.5 0.2] "float roughness" 0.1""",
    "conductor_aniso"  => """Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "float uroughness" 0.02 "float vroughness" 0.4""",

    # Dielectric
    "dielectric"            => """Material "dielectric" "float eta" 1.5""",
    "dielectric_rough"      => """Material "dielectric" "float eta" 1.5 "float uroughness" 0.1 "float vroughness" 0.1""",
    "dielectric_rough_high" => """Material "dielectric" "float eta" 1.5 "float uroughness" 0.3 "float vroughness" 0.3""",
    "dielectric_diamond"    => """Material "dielectric" "float eta" 2.4""",
    "dielectric_bk7"        => """Material "dielectric" "spectrum eta" "glass-BK7" "float roughness" 0.05""",

    # Thin dielectric
    "thindielectric" => """Material "thindielectric" "float eta" 1.5""",

    # Coated diffuse
    "coateddiffuse"       => """Material "coateddiffuse" "rgb reflectance" [0.5 0.3 0.1] "float roughness" 0.1 "float eta" 1.5""",
    "coateddiffuse_rough" => """Material "coateddiffuse" "rgb reflectance" [0.5 0.3 0.1] "float roughness" 0.3 "float eta" 1.5""",
    "coateddiffuse_aniso" => """Material "coateddiffuse" "rgb reflectance" [0.5 0.3 0.1] "float uroughness" 0.05 "float vroughness" 0.3 "float eta" 1.5""",
    # A coat whose IOR is a spectrum: pbrt ends the secondary wavelengths where a path meets it.
    "coateddiffuse_f11"   => """Material "coateddiffuse" "rgb reflectance" [0.5 0.3 0.1] "float roughness" 0.1 "spectrum eta" "glass-F11\"""",

    # Coated conductor
    "coatedconductor"       => """Material "coatedconductor" "spectrum conductor.eta" "metal-Au-eta" "spectrum conductor.k" "metal-Au-k" "float interface.roughness" 0.1 "float conductor.roughness" 0.01""",
    "coatedconductor_rough" => """Material "coatedconductor" "spectrum conductor.eta" "metal-Au-eta" "spectrum conductor.k" "metal-Au-k" "float interface.roughness" 0.3 "float conductor.roughness" 0.1""",
    "coatedconductor_reflectance" => """Material "coatedconductor" "rgb reflectance" [0.85 0.1 0.1] "float interface.roughness" 0.08 "float conductor.roughness" 0.01""",
    "coatedconductor_bk7" => """Material "coatedconductor" "spectrum conductor.eta" "metal-Au-eta" "spectrum conductor.k" "metal-Au-k" "float interface.roughness" 0.1 "float conductor.roughness" 0.01 "spectrum interface.eta" "glass-BK7\"""",

    # Diffuse transmission
    "diffusetransmission" => """Material "diffusetransmission" "rgb reflectance" [0.5 0.5 0.5] "rgb transmittance" [0.5 0.5 0.5]""",

    # Mix
    "mix" => """MakeNamedMaterial "mat_diffuse" "string type" "diffuse" "rgb reflectance" [0.8 0.1 0.1]
MakeNamedMaterial "mat_conductor" "string type" "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "float roughness" 0.01
Material "mix" "string materials" [ "mat_diffuse" "mat_conductor" ] "float amount" 0.5""",
)

# Materials that use texture declarations (checkerboard reflectance)
const TEXTURE_MATERIALS = Dict(
    "diffuse_checker" => (
        textures = """Texture "checker_tex" "spectrum" "checkerboard"
  "rgb tex1" [0.8 0.2 0.1]
  "rgb tex2" [0.1 0.2 0.8]
  "float uscale" 8 "float vscale" 8""",
        material = """Material "diffuse" "texture reflectance" "checker_tex\""""
    ),
    "diffuse_checker_scale4" => (
        textures = """Texture "checker_tex" "spectrum" "checkerboard"
  "rgb tex1" [0.8 0.2 0.1]
  "rgb tex2" [0.1 0.2 0.8]
  "float uscale" 4 "float vscale" 4""",
        material = """Material "diffuse" "texture reflectance" "checker_tex\""""
    ),
    "diffuse_checker_scale16" => (
        textures = """Texture "checker_tex" "spectrum" "checkerboard"
  "rgb tex1" [0.8 0.2 0.1]
  "rgb tex2" [0.1 0.2 0.8]
  "float uscale" 16 "float vscale" 16""",
        material = """Material "diffuse" "texture reflectance" "checker_tex\""""
    ),
    "diffuse_checker_aniso" => (
        textures = """Texture "checker_tex" "spectrum" "checkerboard"
  "rgb tex1" [0.8 0.2 0.1]
  "rgb tex2" [0.1 0.2 0.8]
  "float uscale" 4 "float vscale" 12""",
        material = """Material "diffuse" "texture reflectance" "checker_tex\""""
    ),
    "coateddiffuse_checker" => (
        textures = """Texture "checker_tex" "spectrum" "checkerboard"
  "rgb tex1" [0.6 0.4 0.2]
  "rgb tex2" [0.2 0.4 0.6]
  "float uscale" 8 "float vscale" 8""",
        material = """Material "coateddiffuse" "texture reflectance" "checker_tex" "float roughness" 0.1 "float eta" 1.5"""
    ),
    "conductor_checker_rough" => (
        textures = """Texture "rough_tex" "float" "checkerboard"
  "float tex1" 0.01
  "float tex2" 0.3
  "float uscale" 6 "float vscale" 6""",
        material = """Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "texture roughness" "rough_tex\""""
    ),
    "dielectric_checker_rough" => (
        textures = """Texture "rough_tex" "float" "checkerboard"
  "float tex1" 0.02
  "float tex2" 0.25
  "float uscale" 6 "float vscale" 6""",
        material = """Material "dielectric" "float eta" 1.5 "texture uroughness" "rough_tex" "texture vroughness" "rough_tex\""""
    ),
)

# Filter configurations
const FILTERS = Dict(
    "box"             => "",  # default (no filter directive = pbrt default)
    "gaussian"        => """PixelFilter "gaussian" "float xradius" 1.5 "float yradius" 1.5 "float sigma" 0.5""",
    "gaussian_sharp"  => """PixelFilter "gaussian" "float xradius" 0.75 "float yradius" 0.75 "float sigma" 0.3""",
    "gaussian_wide"   => """PixelFilter "gaussian" "float xradius" 2.5 "float yradius" 2.5 "float sigma" 1.5""",
    "mitchell"        => """PixelFilter "mitchell" "float xradius" 2.0 "float yradius" 2.0""",
    "mitchell_catmull" => """PixelFilter "mitchell" "float xradius" 2.0 "float yradius" 2.0 "float B" 0 "float C" 0.5""",
    "triangle"        => """PixelFilter "triangle" "float xradius" 2.0 "float yradius" 2.0""",
    "triangle_wide"   => """PixelFilter "triangle" "float xradius" 3.0 "float yradius" 3.0""",
    "lanczos"         => """PixelFilter "sinc" "float xradius" 4.0 "float yradius" 4.0 "float tau" 3.0""",
    "lanczos_narrow"  => """PixelFilter "sinc" "float xradius" 2.0 "float yradius" 2.0 "float tau" 3.0""",
)

# Sensor configurations
# name => extra Film parameters. Hikari ships 17 calibrated sensors but only
# nikon_d850 was ever compared against pbrt, so a per-sensor calibration error
# could sit in any of the other 16 unnoticed — crown.pbrt uses
# canon_eos_5d_mkiv at iso 150, a combination nothing exercised.
const SENSORS = Dict(
    "iso200"            => """ "float iso" 200""",
    "iso50"             => """ "float iso" 50""",
    "wb4000"            => """ "float whitebalance" 4000""",   # warm white balance
    "wb8000"            => """ "float whitebalance" 8000""",   # cool white balance
    "nikon_d850"        => """ "string sensor" "nikon_d850\"""",
    "canon_eos_5d_mkiv" => """ "string sensor" "canon_eos_5d_mkiv\"""",
    # crown.pbrt's exact film: this sensor AND a non-default iso together.
    "canon_crown"       => """ "string sensor" "canon_eos_5d_mkiv" "float iso" 150""",
)

# ============================================================================
# Scene writing
# ============================================================================

function write_scene(filepath; camera=CAMERA, film=FILM_BASE, integrator=INTEGRATOR,
                     filter_line="", light="", textures="",
                     ground_mat="""Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]""",
                     sphere_mat="", medium_before="", medium_after="")
    open(filepath, "w") do io
        println(io, "# Auto-generated reference test scene")
        println(io, camera)
        println(io, film)
        !isempty(filter_line) && println(io, filter_line)
        println(io, integrator)
        println(io)
        println(io, "WorldBegin")
        println(io)

        # Medium definitions (before shapes)
        !isempty(medium_before) && (println(io, medium_before); println(io))

        # Light
        println(io, light)
        println(io)

        # Texture definitions
        !isempty(textures) && (println(io, textures); println(io))

        # Ground
        println(io, "# Ground")
        println(io, ground_mat)
        write_mesh(io, GROUND...)
        println(io)

        # Medium interface (if any)
        !isempty(medium_after) && println(io, medium_after)

        # Sphere
        println(io, "# Sphere")
        for line in split(sphere_mat, '\n')
            isempty(strip(line)) || println(io, line)
        end
        write_mesh(io, SPHERE...)
    end
    return filepath
end

# ============================================================================
# Generation functions
# ============================================================================

function generate_material_light_scenes()
    base_lights = ["point", "distant", "spot", "area", "ambient"]
    base_materials = [
        "diffuse", "diffuse_colored",
        "conductor_gold", "conductor_mirror", "conductor_rough",
        "conductor_silver", "conductor_copper", "conductor_reflectance", "conductor_aniso",
        "dielectric", "dielectric_rough", "dielectric_rough_high", "dielectric_diamond", "dielectric_bk7",
        "thindielectric",
        "coateddiffuse", "coateddiffuse_rough", "coateddiffuse_aniso", "coateddiffuse_f11",
        "coatedconductor", "coatedconductor_rough", "coatedconductor_reflectance", "coatedconductor_bk7",
        "diffusetransmission",
    ]

    # Not generated: a scene whose 256-sample render is too noisy for the suite's tile band to
    # say anything. Dispersive glass under the spot light ends every path's secondary wavelengths,
    # and pbrt's own 256-sample image is 0.0575 from its 4096-sample one, so two 256-sample images
    # differ by about 0.08 whatever the renderer. Hikari's matches pbrt's at 4096 samples (tile
    # 0.019, energy 1.001); `test_dispersion.jl` checks the glass's energy under the point light.
    noisy = Set([("dielectric_bk7", "spot")])
    count = 0
    for mat in base_materials, light in base_lights
        (mat, light) in noisy && continue
        name = "mat_$(mat)_light_$(light)"
        write_scene(joinpath(SCENES_DIR, "$(name).pbrt");
                    light=LIGHTS[light], sphere_mat=MATERIALS[mat])
        count += 1
    end

    # Mix (only point light)
    write_scene(joinpath(SCENES_DIR, "mat_mix_light_point.pbrt");
                light=LIGHTS["point"], sphere_mat=MATERIALS["mix"])
    count += 1

    println("  Generated $count material × light scenes")
    return count
end

function generate_texture_scenes()
    count = 0
    for (name, tex_mat) in TEXTURE_MATERIALS
        write_scene(joinpath(SCENES_DIR, "tex_$(name)_light_point.pbrt");
                    light=LIGHTS["point"], textures=tex_mat.textures,
                    sphere_mat=tex_mat.material)
        count += 1
    end
    println("  Generated $count texture scenes")
    return count
end

function generate_light_variant_scenes()
    count = 0
    # Diffuse sphere for all light variants (primary test)
    mat = MATERIALS["diffuse"]
    for (lname, ldef) in LIGHT_VARIANTS
        write_scene(joinpath(SCENES_DIR, "light_$(lname).pbrt");
                    light=ldef, sphere_mat=mat)
        count += 1
    end
    # Conductor (specular) sphere for selected light variants to test specular highlights
    specular_mat = MATERIALS["conductor_gold"]
    for lname in ["point_warm", "point_side", "point_top", "two_points", "area_large"]
        write_scene(joinpath(SCENES_DIR, "light_$(lname)_specular.pbrt");
                    light=LIGHT_VARIANTS[lname], sphere_mat=specular_mat)
        count += 1
    end
    println("  Generated $count light variant scenes")
    return count
end

function generate_filter_scenes()
    count = 0
    mat = MATERIALS["diffuse"]
    light = LIGHTS["point"]
    for (fname, fline) in FILTERS
        write_scene(joinpath(SCENES_DIR, "filter_$(fname).pbrt");
                    light=light, sphere_mat=mat, filter_line=fline)
        count += 1
    end
    println("  Generated $count filter scenes")
    return count
end

function generate_sensor_scenes()
    count = 0
    mat = MATERIALS["conductor_gold"]
    light = LIGHTS["point"]
    for (sname, film_params) in SENSORS
        write_scene(joinpath(SCENES_DIR, "sensor_$(sname).pbrt");
                    light=light, sphere_mat=mat, film=FILM_BASE * film_params)
        count += 1
    end
    println("  Generated $count sensor scenes")
    return count
end

function generate_medium_scenes()
    count = 0
    light = LIGHTS["point"]

    # NanoVDB cloud: generate Perlin/Worley density, save as .nvdb, reference in scene
    # Sphere is at center (0,0,0.5) radius 0.5 → NanoVDB bounds match
    nvdb_path = joinpath(SCENES_DIR, "cloud_density.nvdb")
    cloud_data = Hikari.generate_cloud_density(128; scale=2.5, threshold=0.15, worley_weight=0.2, edge_sharpness=4.0, density_scale=4.5)
    Hikari.save_nanovdb(nvdb_path, cloud_data, (-0.5, -0.5, 0.0), (1.0, 1.0, 1.0))

    nvdb_medium_def = """MakeNamedMedium "cloud"
  "string type" "nanovdb"
  "string filename" "cloud_density.nvdb"
  "rgb sigma_a" [0.5 0.5 0.5]
  "rgb sigma_s" [15.0 15.0 15.0]
  "float g" 0.0"""
    medium_interface = """MediumInterface "cloud" \"\""""
    sphere_mat = """Material "dielectric" "float eta" 1.0"""

    cloud_light = LIGHTS["point"] * "\nLightSource \"infinite\" \"rgb L\" [0.5 0.5 0.5]"
    write_scene(joinpath(SCENES_DIR, "medium_cloud_point.pbrt");
                light=cloud_light, sphere_mat=sphere_mat,
                medium_before=nvdb_medium_def, medium_after=medium_interface)
    count += 1

    # Colored absorbing medium (tinted glass)
    medium_def2 = """MakeNamedMedium "tinted"
  "string type" "homogeneous"
  "rgb sigma_a" [0.1 0.5 0.1]
  "rgb sigma_s" [0.0 0.0 0.0]
  "float g" 0.0"""
    medium_interface2 = """MediumInterface "" "tinted\""""
    sphere_mat2 = """Material "dielectric" "float eta" 1.5"""

    write_scene(joinpath(SCENES_DIR, "medium_tinted_point.pbrt");
                light=light, sphere_mat=sphere_mat2,
                medium_before=medium_def2, medium_after=medium_interface2)
    count += 1

    # Milk (Wholemilk preset, scale=0.1) — dielectric glass sphere
    milk_def = """MakeNamedMedium "milk"
  "string type" "homogeneous"
  "rgb sigma_s" [0.255 0.321 0.377]
  "rgb sigma_a" [0.00011 0.00024 0.0014]
  "float g" 0.0"""
    write_scene(joinpath(SCENES_DIR, "medium_milk_point.pbrt");
                light=light,
                sphere_mat="""Material "dielectric" "float eta" 1.5""",
                medium_before=milk_def,
                medium_after="""MediumInterface "milk" \"\"""")
    count += 1

    # Coffee (Espresso preset, scale=0.5) — dielectric glass sphere
    coffee_def = """MakeNamedMedium "coffee"
  "string type" "homogeneous"
  "rgb sigma_s" [0.36 0.425 0.51]
  "rgb sigma_a" [2.4 3.29 4.425]
  "float g" 0.0"""
    write_scene(joinpath(SCENES_DIR, "medium_coffee_point.pbrt");
                light=light,
                sphere_mat="""Material "dielectric" "float eta" 1.5""",
                medium_before=coffee_def,
                medium_after="""MediumInterface "coffee" \"\"""")
    count += 1

    # Smoke (density=5.0, albedo=0.95, g=0.3) — thin dielectric shell (eta=1)
    smoke_def = """MakeNamedMedium "smoke"
  "string type" "homogeneous"
  "rgb sigma_s" [4.75 4.75 4.75]
  "rgb sigma_a" [0.25 0.25 0.25]
  "float g" 0.3"""
    write_scene(joinpath(SCENES_DIR, "medium_smoke_point.pbrt");
                light=light,
                sphere_mat="""Material "dielectric" "float eta" 1.0""",
                medium_before=smoke_def,
                medium_after="""MediumInterface "smoke" \"\"""")
    count += 1

    println("  Generated $count medium scenes")
    return count
end

"""
Thin-lens depth of field.

pbrt's camera default is a pinhole (`lensradius` 0), so a renderer that parses
the scene but never applies `lensradius`/`focaldistance` still gets the ENERGY
right — it just renders everything sharp. That failure is invisible to an
energy-ratio gate and only shows up in the spatial tile score, which is why it
needs its own scene rather than a knob on an existing one.

Three quads stacked along Z; the camera focuses on the far one, so the near and
middle quads must come out visibly blurred.
"""
function generate_camera_scenes()
    # Camera sits at (0,-5,0) looking along +y (suite convention), so the quads
    # must lie in the XZ plane with normal -y to FACE it. Putting them in XY with
    # normal +z renders them edge-on: the image comes out black and the scene
    # then "passes" against an equally black reference while testing nothing.
    #
    # Depths from the camera are 3 / 5 / 7; each quad is offset in x so all three
    # stay visible instead of hiding behind each other. A checkerboard gives the
    # high-frequency detail that makes defocus blur measurable — a flat colour
    # blurs into itself and barely moves the tile score.
    function quad(io, xoff, yoff, tex)
        println(io, """
AttributeBegin
Material "diffuse" "texture reflectance" "$tex"
Translate $xoff $yoff 0
Shape "trianglemesh"
  "point3 P" [ -0.45 0 -0.45  0.45 0 -0.45  0.45 0 0.45  -0.45 0 0.45 ]
  "normal N" [ 0 -1 0  0 -1 0  0 -1 0  0 -1 0 ]
  "point2 uv" [ 0 0  1 0  1 1  0 1 ]
  "integer indices" [ 0 1 2  0 2 3 ]
AttributeEnd""")
    end

    n = 0
    for (name, lensradius, focaldistance) in (("cam_dof_near", 0.3, 3.0),
                                              ("cam_dof_far",  0.3, 7.0),
                                              ("cam_pinhole",  0.0, 5.0))
        path = joinpath(SCENES_DIR, "$(name)_light_point.pbrt")
        open(path, "w") do io
            println(io, "# Auto-generated reference test scene — thin-lens depth of field.")
            println(io, "LookAt 0 -5 0   0 0 0   0 0 1")
            println(io, """Camera "perspective" "float fov" 50""")
            println(io, """    "float lensradius" $lensradius""")
            println(io, """    "float focaldistance" $focaldistance""")
            println(io, FILM_BASE)
            println(io, INTEGRATOR)
            println(io)
            println(io, "WorldBegin")
            # Uniform illumination: a point light would land behind these quads.
            println(io, """LightSource "infinite" "rgb L" [1 1 1]""")
            println(io)
            for (i, (c1, c2)) in enumerate((("0.9 0.15 0.1", "0.05 0.05 0.05"),
                                            ("0.1 0.9 0.15", "0.05 0.05 0.05"),
                                            ("0.15 0.2 0.9", "0.05 0.05 0.05")))
                println(io, """Texture "checks$i" "spectrum" "checkerboard" "float uscale" 6 "float vscale" 6 "rgb tex1" [$c1] "rgb tex2" [$c2]""")
            end
            println(io)
            quad(io, -0.8, -2.0, "checks1")   # distance 3
            quad(io,  0.0,  0.0, "checks2")   # distance 5
            quad(io,  0.8,  2.0, "checks3")   # distance 7
        end
        n += 1
    end
    return n
end

"""
Integrator defaults that pbrt-v4 and Hikari DISAGREE on.

Every other scene in this suite is written with `INTEGRATOR`, which pins
`"bool regularize" true` and a `maxcomponentvalue` of 10 — and those happen to
be Hikari's own defaults, so the whole suite is blind to the disagreement:

    regularize         pbrt false (cpu/integrators.cpp:817)   Hikari true
    maxcomponentvalue  pbrt Infinity (film.cpp:576)           Hikari 10f0

Both roughen or clamp specular highlights. A scene that simply omits the knobs
— which is what real scenes do — therefore renders with energy pushed out of
its highlights unless the loader threads the parsed values through. Measured on
crown.pbrt that was a 1.6 % global energy deficit sitting entirely in the bright
quintiles.

These use a smooth conductor, where regularization and clamping actually bite.
"""
function generate_integrator_scenes()
    n = 0
    for (name, integ, film) in (
            # pbrt defaults on both knobs: no regularization, no clamp.
            ("integ_default_smooth_conductor",
             """Integrator "volpath" "integer maxdepth" 8""",
             """Film "rgb" "integer xresolution" $RES "integer yresolution" $RES"""),
            # Regularization explicitly ON, clamp still off — isolates one knob.
            ("integ_regularize_smooth_conductor",
             """Integrator "volpath" "integer maxdepth" 8 "bool regularize" true""",
             """Film "rgb" "integer xresolution" $RES "integer yresolution" $RES"""),
            # Clamp explicitly on, regularization off — isolates the other.
            ("integ_clamp_smooth_conductor",
             """Integrator "volpath" "integer maxdepth" 8""",
             """Film "rgb" "integer xresolution" $RES "integer yresolution" $RES "float maxcomponentvalue" 2"""))
        path = joinpath(SCENES_DIR, "$(name)_light_area.pbrt")
        write_scene(path; film=film, integrator=integ, light=LIGHTS["area"],
                    sphere_mat=MATERIALS["conductor_mirror"])
        n += 1
    end
    return n
end

"""
Scenes that differ ONLY in how many area lights they contain.

Every other scene in this suite has exactly one light, so light SELECTION — the
sampling PMF over lights and the MIS weights that ride on it — is never
exercised. A per-light bias is invisible with one light and grows with the
count. crown.pbrt has six area lights and reads ~2 % brighter than pbrt, which
is the shape such a bias would take.

Per-light emission is held FIXED and each scene is compared against its own pbrt
reference, so the diagnostic is whether the energy ratio DRIFTS as lights are
added, not the absolute brightness.

Blackbody + `scale`, matching how crown's lights are written.
"""
function generate_multilight_scenes()
    placements = ((0.8, -0.4, 1.5), (-0.9, -0.3, 1.4), (0.6, 0.9, 1.6), (-0.7, 0.8, 1.3),
                  (0.0, -1.1, 1.2), (0.1, 1.2, 1.7))
    emitter((x, y, z)) = """
AttributeBegin
AreaLightSource "diffuse" "float scale" [10] "blackbody L" [5500] "bool twosided" true
Material "diffuse" "rgb reflectance" [1 1 1]
Translate $x $y $z
Shape "trianglemesh"
  "point3 P" [ -0.25 -0.25 0  0.25 -0.25 0  0.25 0.25 0  -0.25 0.25 0 ]
  "integer indices" [ 0 1 2  0 2 3 ]
AttributeEnd"""

    n = 0
    for count in (1, 2, 4, 6)
        write_scene(joinpath(SCENES_DIR, "multilight_$(lpad(count, 2, '0'))_conductor.pbrt");
                    light=join((emitter(p) for p in placements[1:count]), "\n"),
                    sphere_mat=MATERIALS["conductor_gold"])
        n += 1
    end
    return n
end

"""
crown.pbrt's `mitra_right_back` material, reproduced on the suite sphere, plus
the FLAT control that localises bump bugs to surface curvature.

`tex_conductor_bumpmap_*` already covers an image bump, but with a 16-bit gray
map whose height span is 0.144 and whose largest texel-to-texel step is 0.0024 —
a nearly flat height field. crown's map spans 0.737 with steps of 0.047, 5x and
20x larger, and rides on a `scale` indirection over a material that also has a
textured roughness. A gentle bump hides errors that scale with amplitude.

The flat/curved PAIR is the diagnostic. pbrt's eq. 9.20 adds `h * dndu`, and the
shading frame it feeds the formula is dpdu PROJECTED perpendicular to the shading
normal. Both terms vanish on a flat surface, where the interpolated normal is
already perpendicular to dpdu. So a bump bug that is curvature-coupled shows up
as: flat matches, sphere does not. That is exactly how the raw-dpdu bug was
found — flat 0.0008 / 0.9999 against sphere 0.4091 / 0.918.
"""
function generate_crownlike_bump_scenes()
    textures = """
Texture "bump-raw" "float" "imagemap"
    "string filename" "textures/test_bump_rgba.png"
Texture "bump-sc" "float" "scale"
    "float scale" 0.25
    "texture tex" "bump-raw"
Texture "rough-raw" "float" "imagemap"
    "string filename" "textures/test_midgrey_stripes.png"
Texture "rough-sc" "float" "scale"
    "float scale" 0.1
    "texture tex" "rough-raw\""""

    # Gold conductor with BOTH textures, exactly as crown declares it.
    mat = """Material "conductor"
    "spectrum eta" "metal-Au-eta"
    "spectrum k" "metal-Au-k"
    "texture roughness" "rough-sc"
    "texture displacement" "bump-sc\""""

    # Control: same bump, but wired straight to the material with a constant
    # roughness. If the scaled/textured variant misses and this one does not,
    # the indirection is implicated rather than the bump image itself.
    mat_direct = """Material "conductor"
    "spectrum eta" "metal-Au-eta"
    "spectrum k" "metal-Au-k"
    "float roughness" 0.05
    "texture displacement" "bump-raw\""""

    n = 0
    for (suffix, light) in (("light_point", LIGHTS["point"]), ("light_area", LIGHTS["area"]))
        write_scene(joinpath(SCENES_DIR, "tex_conductor_scalebump_texrough_$(suffix).pbrt");
                    light = light, textures = textures, sphere_mat = mat)
        write_scene(joinpath(SCENES_DIR, "tex_conductor_rgbabump_direct_$(suffix).pbrt");
                    light = light, textures = textures, sphere_mat = mat_direct)
        n += 2
    end

    # Flat control: same map, amplitude and material, zero curvature. Written by
    # hand because `write_scene` always emits the ground + sphere pair.
    open(joinpath(SCENES_DIR, "tex_conductor_bump_flat_light_point.pbrt"), "w") do io
        println(io, "# Auto-generated reference test scene — FLAT bump control.")
        println(io, "# dndu = dndv = 0 here and the shading frame needs no projection,")
        println(io, "# so a curvature-coupled bump bug matches on this and misses on the")
        println(io, "# sphere. Keep the two together: the PAIR is what localises the bug.")
        println(io, CAMERA); println(io, FILM_BASE); println(io, INTEGRATOR)
        println(io); println(io, "WorldBegin"); println(io, LIGHTS["point"]); println(io)
        println(io, textures); println(io)
        for line in split(mat_direct, '\n'); println(io, line); end
        println(io, """Shape "trianglemesh"
  "point3 P" [ -0.5 0 0.0   0.5 0 0.0   0.5 0 1.0   -0.5 0 1.0 ]
  "normal N" [ 0 -1 0  0 -1 0  0 -1 0  0 -1 0 ]
  "point2 uv" [ 0 0  1 0  1 1  0 1 ]
  "integer indices" [ 0 1 2  0 2 3 ]""")
    end
    return n + 1
end

function generate_all_scenes()
    println("Generating scenes in $SCENES_DIR...")

    # Clear old scenes
    for f in readdir(SCENES_DIR)
        endswith(f, ".pbrt") && rm(joinpath(SCENES_DIR, f))
    end

    total = 0
    total += generate_material_light_scenes()
    total += generate_texture_scenes()
    total += generate_light_variant_scenes()
    total += generate_filter_scenes()
    total += generate_sensor_scenes()
    total += generate_medium_scenes()
    total += generate_camera_scenes()
    total += generate_integrator_scenes()
    total += generate_multilight_scenes()
    total += generate_crownlike_bump_scenes()

    println("Total: $total scene files")
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_all_scenes()
end
