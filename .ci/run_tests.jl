module HikariCITests
    include(joinpath(@__DIR__, "..", "test", "test_files.jl"))

    # The unit files, dealt round-robin into `groups` processes, or the pbrt reference suite.
    function test_files(group::String; groups::Int = 3)
        startswith(group, "pbrt") && return ["pbrt"]
        index = parse(Int, group)
        1 <= index <= groups || throw(ArgumentError("test group must be between 1 and $groups, or pbrt[-category...]"))
        return TEST_FILES[index:groups:end]
    end

    # `pbrt-textures-other` runs those two categories of the reference suite, `pbrt` all of them.
    pbrt_categories(group::String) =
        group == "pbrt" ? "materials,textures,other" : replace(chopprefix(group, "pbrt-"), "-" => ",")

    # In this project, the environment `setup.jl` resolved and precompiled once for every job:
    # a `Pkg.test` sandbox is another environment and precompiled again. A fresh process per
    # group, with this process's flags, which are those the depot was precompiled for.
    function run_tests(group::String)
        cmd = `$(Base.julia_cmd()) --project=$(@__DIR__) runtests.jl $(test_files(group))`
        if startswith(group, "pbrt")
            cmd = addenv(cmd, "HIKARI_PBRT_CATEGORIES" => pbrt_categories(group))
        end
        run(Cmd(cmd; dir = joinpath(dirname(@__DIR__), "test")))
        return nothing
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    HikariCITests.run_tests(only(ARGS))
end
