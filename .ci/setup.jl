using Pkg, TOML

Pkg.activate(@__DIR__)
Pkg.instantiate()

# The docs and comparison gallery use the same backend revisions as the tests.
# Only their additional registered dependencies are added here.
for project in ARGS
    dependencies = TOML.parsefile(joinpath(@__DIR__, "..", project, "Project.toml"))["deps"]
    installed = Pkg.project().dependencies
    additions = [
        PackageSpec(; name, uuid) for (name, uuid) in dependencies
            if !haskey(installed, name)
    ]
    isempty(additions) || Pkg.add(additions)
end
