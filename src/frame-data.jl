using CodecZstd: ZstdDecompressorStream, ZstdCompressorStream
using Serialization: serialize, deserialize

function load_frame_data(path::AbstractString)
    open(path,"r") do io
        stream = ZstdDecompressorStream(io)
        try
            data = deserialize(stream)
            eof(stream) || error("Trailing bytes in compressed frame data")
            return data
        finally
            close(stream)
        end
    end
end

function save_frame_data(path::AbstractString, data)
    mkpath(dirname(abspath(path)))
    temporary, io = mktemp(dirname(abspath(path)))
    try
        stream = ZstdCompressorStream(io)
        try
            serialize(stream,data)
        finally
            close(stream)
        end
        mv(temporary,path;force=true)
        return path
    catch
        close(io)
        isfile(temporary) && rm(temporary)
        rethrow()
    end
end
