using Test, Hikari, Lava

function read_cie_curves!(out, x, y, z)
    table = Hikari.CIEXYZTable(x, y, z)
    @inbounds out[1] = table.cie_x[1] + table.cie_y[1] + table.cie_z[1]
    return nothing
end

@testset "CIE curves accept independently read-only arguments" begin
    D = Lava.LavaDeviceArray{Float32,1}
    C = Lava.Const{Float32,1}
    # Every combination occurs in an otherwise identical reconstruction of
    # the CIE table. A shared field type rejects mixed Const/device arrays.
    for X in (D, C), Y in (D, C), Z in (D, C)
        kernel = Lava.lava_compile_gpu(read_cie_curves!, Tuple{D,X,Y,Z})
        @test !isempty(kernel.spirv_bytes)
    end
end
