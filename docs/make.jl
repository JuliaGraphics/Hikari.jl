using Documenter
using DocumenterVitepress
using Hikari

makedocs(; sitename = "Hikari", authors = "Anton Smirnov, Simon Danisch and contributors",
    modules = [Hikari],
    checkdocs = :all,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/JuliaGraphics/Hikari.jl", # this must be the full URL!
        devbranch = "master",
        devurl = "dev";
    ),
    draft = false,
    source = "src",
    build = "build",
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "Get Started" => "get_started.md",
        "Shadows" => "shadows.md",
        "Materials" => "materials.md",
        "API" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/JuliaGraphics/Hikari.jl",
    target = "build", # this is where Vitepress stores its output
    branch = "gh-pages",
    devbranch = "master",
    push_preview = true,
)
