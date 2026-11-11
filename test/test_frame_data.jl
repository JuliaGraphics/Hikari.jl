using Test, Hikari

@testset "Lossless file-backed frame data" begin
    mktempdir() do directory
        path=joinpath(directory,"frame.jls.zst")
        values=Float32[0,-0.0,Inf,-Inf,NaN,1/3]
        data=(;time=157/24,values,shared=values)
        @test Hikari.save_frame_data(path,data)==path
        loaded=Hikari.load_frame_data(path)
        @test loaded.time==data.time
        @test reinterpret(UInt32,loaded.values)==reinterpret(UInt32,data.values)
        @test loaded.values===loaded.shared
        replacement=(;time=158/24,values=reverse(values))
        Hikari.save_frame_data(path,replacement)
        @test Hikari.load_frame_data(path).time==158/24
        original=read(path)
        write(path,original[1:length(original)÷2])
        @test_throws Exception Hikari.load_frame_data(path)
    end
end
