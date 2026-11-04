# ============================================================================
# PBRT file tokenizer
# ============================================================================
# Tokenizes pbrt-v4 scene description files into a stream of tokens.
# The format is simple: whitespace-separated tokens with quoted strings,
# brackets, comments (#), numbers, and directive words.

@enum TokenType begin
    TOK_STRING    # "quoted string"
    TOK_NUMBER    # 1.5, -3.2e-4, 42
    TOK_LBRACKET  # [
    TOK_RBRACKET  # ]
    TOK_WORD      # WorldBegin, Shape, etc.
end

struct Token
    type::TokenType
    value::String
    line::Int
end

function tokenize(text::AbstractString)
    tokens = Token[]
    i = firstindex(text)
    line = 1
    n = lastindex(text)

    while i <= n
        c = text[i]

        # Newlines
        if c == '\n'
            line += 1
            i = nextind(text, i)
            continue
        end

        # Whitespace
        if c == ' ' || c == '\t' || c == '\r'
            i = nextind(text, i)
            continue
        end

        # Comments — skip to end of line
        if c == '#'
            while i <= n && text[i] != '\n'
                i = nextind(text, i)
            end
            continue
        end

        # Brackets
        if c == '['
            push!(tokens, Token(TOK_LBRACKET, "[", line))
            i = nextind(text, i)
            continue
        end
        if c == ']'
            push!(tokens, Token(TOK_RBRACKET, "]", line))
            i = nextind(text, i)
            continue
        end

        # Quoted string
        if c == '"'
            i = nextind(text, i)
            buf = IOBuffer()
            while i <= n && text[i] != '"'
                if text[i] == '\\'
                    i = nextind(text, i)
                    if i <= n
                        c2 = text[i]
                        if c2 == 'n'
                            write(buf, '\n')
                        elseif c2 == 't'
                            write(buf, '\t')
                        elseif c2 == '\\'
                            write(buf, '\\')
                        elseif c2 == '"'
                            write(buf, '"')
                        else
                            write(buf, c2)
                        end
                    end
                else
                    text[i] == '\n' && (line += 1)
                    write(buf, text[i])
                end
                i = nextind(text, i)
            end
            i <= n && (i = nextind(text, i))  # skip closing "
            push!(tokens, Token(TOK_STRING, String(take!(buf)), line))
            continue
        end

        # Number or word — read until delimiter
        start = i
        while i <= n
            ci = text[i]
            (ci == ' ' || ci == '\t' || ci == '\r' || ci == '\n' ||
             ci == '[' || ci == ']' || ci == '"' || ci == '#') && break
            i = nextind(text, i)
        end
        word = SubString(text, start, prevind(text, i))

        if tryparse(Float64, word) !== nothing
            push!(tokens, Token(TOK_NUMBER, String(word), line))
        else
            push!(tokens, Token(TOK_WORD, String(word), line))
        end
    end

    return tokens
end

# Token stream for the parser
mutable struct TokenStream
    tokens::Vector{Token}
    pos::Int
    filename::String
    base_dir::String
end

TokenStream(tokens::Vector{Token}; filename="<string>", base_dir=".") =
    TokenStream(tokens, 1, filename, base_dir)

Base.eof(ts::TokenStream) = ts.pos > length(ts.tokens)

function peek(ts::TokenStream)
    eof(ts) && return nothing
    return ts.tokens[ts.pos]
end

function next!(ts::TokenStream)
    eof(ts) && error("unexpected end of file in $(ts.filename)")
    t = ts.tokens[ts.pos]
    ts.pos += 1
    return t
end

function expect!(ts::TokenStream, type::TokenType)
    t = next!(ts)
    t.type == type || error("$(ts.filename):$(t.line): expected $(type), got $(t.type) '$(t.value)'")
    return t
end
