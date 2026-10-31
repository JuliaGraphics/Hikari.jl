# Diagnostic combination: common reflection geometry and inline evaluation.
# Include shared_reflection.jl into Hikari once before loading this file.
let source = read(joinpath(@__DIR__, "shared_reflection.jl"), String)
    first = findfirst("Base.@constprop :none @noinline @generated function surface_apply_fields", source).start
    method = replace(source[first:end], "@noinline" => "@inline", ":noinline" => ":inline")
    Base.include_string(@__MODULE__, method, joinpath(@__DIR__, "inline_shared_reflection_method.jl"))
end
