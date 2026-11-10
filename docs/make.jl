using Documenter
using DocumenterVitepress
using Hikari

# `docs_image` and `docs_samples` for the pages, and the check that no image is stale.
include(joinpath(@__DIR__, "renders.jl"))

# The comparison page links to `pbrt-comparison/gallery.html`. CI adds the gallery when it
# deploys, built from the pbrt test groups' records; a local build copies the one in
# `test/pbrt/display` (`test/pbrt/run_comparison.jl`) if there is one.
let display = joinpath(@__DIR__, "..", "test", "pbrt", "display")
    if isfile(joinpath(display, "gallery.html"))
        gallery = mkpath(joinpath(@__DIR__, "src", "public", "pbrt-comparison"))
        for f in readdir(display)
            (endswith(f, ".png") || f == "gallery.html") && cp(joinpath(display, f), joinpath(gallery, f); force = true)
        end
    end
end

# ============================================================================
# Build docs
# ============================================================================

check_images(joinpath(@__DIR__, "src"))

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
        # One menu in the top bar: ten pages side by side ran past the edge of the window.
        "Guide" => [
            "Rendering" => "shadows.md",
            "Materials" => "materials.md",
            "Textures and Node Graphs" => "textures.md",
            "Material Recipes" => "recipes.md",
            "Lights" => "lights.md",
            "Postprocessing and Denoising" => "postprocessing.md",
        ],
        "pbrt-v4 Comparison" => "comparison.md",
        "API" => "api.md",
    ],
)

# CI deploys separately, after adding the comparison gallery (.github/workflows/ci.yml).
