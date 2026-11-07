# Diagnostic: remove only the shared evaluation function boundary, retaining
# runtime program setup and the same stored records and scattering formulas.
let source = read(joinpath(@__DIR__, "program.jl"), String)
    first = findfirst("Base.@constprop :none @noinline @generated function surface_apply_fields", source).start
    after = findnext("\nsubsurface_slot", source, first).start
    method = replace(source[first:prevind(source, after)], "@noinline" => "@inline", ":noinline" => ":inline")
    Base.include_string(@__MODULE__, method, joinpath(@__DIR__, "inline_evaluation_method.jl"))
end
