# Material Interface Implementation

# Whether a material emits light: an `Emission`, whose faces are area lights.
@propagate_inbounds is_emissive(::Material) = false

# ============================================================================
# Descriptions: the materials a user writes
# ============================================================================

"""
    @description struct Name <: Material
        field = default
        …
    end

A material as a user writes it: a struct with a type parameter per field, so each field holds what
it was given (a number, a colour, an image or a node graph), and a keyword constructor with the
defaults. Hikari turns it into what it renders with when it is pushed (`surface_material`).
"""
macro description(ex)
    (ex isa Expr && ex.head === :struct) || error("@description takes a struct definition")
    head = ex.args[2]
    name, super = head isa Expr && head.head === :<: ? (head.args[1], head.args[2]) : (head, :Material)
    body = filter(a -> a isa Expr && a.head === :(=), ex.args[3].args)
    fields = [a.args[1] for a in body]
    params = [Symbol(uppercasefirst(string(f)), :T) for f in fields]
    typed = [:($f::$p) for (f, p) in zip(fields, params)]
    keywords = [Expr(:kw, f, a.args[2]) for (f, a) in zip(fields, body)]
    return esc(quote
        Base.@__doc__ struct $name{$(params...)} <: $super
            $(typed...)
        end
        $name(; $(keywords...)) = $name($(fields...))
    end)
end

"""
`m` with its field `name` set to `value`, every other field as it was. Rebuilt through the type's
name, not its concrete type, so the field takes the type of `value`.
"""
function with_field(m::M, name::Symbol, value) where {M <: Material}
    values = map(f -> f === name ? value : getfield(m, f), fieldnames(M))
    return Base.typename(M).wrapper(values...)
end
