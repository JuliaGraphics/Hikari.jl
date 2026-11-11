using Pkg, TOML, Dates

# One environment for every CI job: the test groups, the pbrt comparison, the docs and their
# deployment. The setup job resolves and precompiles it once and saves the depot; every other
# job restores that depot and this project and never runs Pkg itself.
const HIKARI = dirname(@__DIR__)

# What the jobs load beyond this project: what `Pkg.test` would make loadable (Hikari's
# dependencies and its test target), and the docs' and the pbrt comparison's dependencies.
function requirements(hikari = TOML.parsefile(joinpath(HIKARI, "Project.toml")))
    test = Dict(name => hikari["extras"][name] for name in hikari["targets"]["test"])
    deps = merge(hikari["deps"], test)
    for project in ("docs", joinpath("test", "pbrt"))
        merge!(deps, TOML.parsefile(joinpath(HIKARI, project, "Project.toml"))["deps"])
    end
    return deps
end

Pkg.activate(@__DIR__)
# Precompile once, after everything is added, not after each operation.
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
Pkg.instantiate()
installed = Pkg.project().dependencies
additions = [PackageSpec(; name, uuid) for (name, uuid) in requirements() if !haskey(installed, name)]
isempty(additions) || Pkg.add(additions)

# Lazy artifacts too (OIDN), so no job downloads them on its own.
artifacts = joinpath(HIKARI, "Artifacts.toml")
for name in keys(TOML.parsefile(artifacts))
    Pkg.Artifacts.ensure_artifact_installed(name, artifacts)
end

Pkg.precompile(; strict = true)
# The depot was restored from an earlier run: drop the package versions and artifacts this
# manifest no longer uses, or the saved cache grows with every run.
Pkg.gc(; collect_delay = Day(0))
