using Documenter
using DocumenterVitepress
using Hikari

# ============================================================================
# Generate comparison.md (gallery injected by CI or from local build)
# ============================================================================

const DOCS_DIR    = @__DIR__
const DISPLAY_DIR = joinpath(DOCS_DIR, "..", "test", "pbrt", "display")
const ASSETS_CMP  = joinpath(DOCS_DIR, "src", "public", "pbrt-comparison")
mkpath(ASSETS_CMP)

# Copy gallery from test/pbrt/display/ if available (local builds, or CI pre-step)
n_scenes = 0
if isdir(DISPLAY_DIR) && isfile(joinpath(DISPLAY_DIR, "gallery.html"))
    for f in readdir(DISPLAY_DIR)
        (endswith(f, ".png") || f == "gallery.html") || continue
        cp(joinpath(DISPLAY_DIR, f), joinpath(ASSETS_CMP, f); force=true)
    end
    n_scenes = count(f -> startswith(f, "ref_") && endswith(f, ".png"), readdir(DISPLAY_DIR))
    println("Comparison: copied gallery + PNGs for $n_scenes scenes")
else
    @warn "No gallery found in $DISPLAY_DIR -- comparison page will be empty"
end

open(joinpath(DOCS_DIR, "src", "comparison.md"), "w") do io
    print(io, """## Hikari vs pbrt-v4 Comparison

Hikari is validated against [pbrt-v4](https://github.com/mmp/pbrt-v4) across $(n_scenes) test scenes covering all material types, light types, and rendering configurations.
Each scene is rendered at 256 samples per pixel.

The **tile score** is a perceptual difference metric (log-space L2 over 16x16 tiles, p95). The **energy ratio** is total scene luminance Hikari / pbrt-v4; values outside 0.95-1.05 are highlighted in red.

```@raw html
<a href="/pbrt-comparison/gallery.html" target="_blank"
   style="display:inline-block; padding:12px 24px; background:#4CAF50; color:white;
   text-decoration:none; border-radius:6px; font-weight:bold; font-size:16px;">
   Open comparison gallery ($(n_scenes) scenes)
</a>
```
""")
end

# ============================================================================
# Build docs
# ============================================================================

makedocs(; sitename = "Hikari", authors = "Anton Smirnov, Simon Danisch and contributors",
    modules = [Hikari],
    checkdocs = :exports,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/JuliaGraphics/Hikari.jl",
        devbranch = "master",
        devurl = "dev";
    ),
    draft = false,
    source = "src",
    build = "build",
    warnonly = false,
    pages = [
        "Home" => "index.md",
        "Get Started" => "get_started.md",
        "Rendering" => "shadows.md",
        "Materials" => "materials.md",
        "Textures and Node Graphs" => "textures.md",
        "Material Recipes" => "recipes.md",
        "Lights" => "lights.md",
        "Postprocessing and Denoising" => "postprocessing.md",
        "pbrt-v4 Comparison" => "comparison.md",
        "API" => "api.md",
    ],
)

# Deployment is handled separately by CI (which injects the pbrt comparison
# gallery into the build before deploying). See .github/workflows/ci.yml.
