module HikariCITests
    using Pkg
    include(joinpath(@__DIR__, "..", "test", "test_files.jl"))

    function test_group(group::String; groups::Int = 4)
        group == "pbrt" && return ["pbrt"]
        group in ("pbrt-materials", "pbrt-textures", "pbrt-other") && return ["pbrt"]
        index = parse(Int, group)
        1 <= index <= groups || throw(ArgumentError("test group must be between 1 and $groups, or pbrt"))
        return TEST_FILES[index:groups:end]
    end

    function run_tests(group::String)
        category = startswith(group, "pbrt-") ? group[6:end] : "all"
        return withenv("HIKARI_PBRT_GROUP" => category) do
            Pkg.test(
                "Hikari"; allow_reresolve = false,
                test_args = test_group(group), julia_args = `--check-bounds=auto`
            )
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    HikariCITests.run_tests(only(ARGS))
end
