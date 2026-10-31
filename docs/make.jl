using Documenter
using DocumenterVitepress
using Trace

makedocs(; sitename = "Trace", authors = "Anton Smirnov and contributors",
    modules = [Trace],
    checkdocs = :all,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/pxl-th/Trace.jl", # this must be the full URL!
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
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/pxl-th/Trace.jl",
    target = "build", # this is where Vitepress stores its output
    branch = "gh-pages",
    devbranch = "master",
    push_preview = true,
)
