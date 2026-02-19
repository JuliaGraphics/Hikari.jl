# RGBToSpectrumTable GPU Struct Passing Test
#
# Tests that RGBToSpectrumTable can be passed directly to OpenCL kernels
# without decomposing into individual fields (res, scale, coeffs).

using Test
using Hikari: RGBToSpectrumTable, get_srgb_table, rgb_to_spectrum, RGBSigmoidPolynomial, to_gpu
using KernelAbstractions
using KernelAbstractions: @kernel, @index
import KernelAbstractions as KA
using Adapt
using pocl_jll
using OpenCL

# Select pocl CPU device for testing
function get_pocl_backend()
    pocl_platform = OpenCL.cl.platforms()[1]  # Portable Computing Language
    pocl_device = OpenCL.cl.devices(pocl_platform)[1]
    OpenCL.cl.device!(pocl_device)
    return OpenCL.OpenCLBackend()
end

# ============================================================================
# Test Kernel - Pass RGBToSpectrumTable as a single struct argument
# ============================================================================

@kernel function rgb2spec_table_test_kernel!(
    output_c0, output_c1, output_c2,
    @Const(table),  # Pass the whole RGBToSpectrumTable struct
    @Const(r), @Const(g), @Const(b)
)
    i = @index(Global)
    @inbounds begin
        poly = rgb_to_spectrum(table, r[i], g[i], b[i])
        output_c0[i] = poly.c0
        output_c1[i] = poly.c1
        output_c2[i] = poly.c2
    end
end

# ============================================================================
# Tests
# ============================================================================

@testset "RGBToSpectrumTable GPU Struct Passing" begin
    backend = get_pocl_backend()

    # Load the sRGB table and convert to GPU
    cpu_table = get_srgb_table()
    gpu_table = to_gpu(backend, cpu_table)

    @testset "Table struct can be adapted to GPU arrays" begin
        @test gpu_table.res == cpu_table.res
        @test gpu_table.scale isa OpenCL.CLArray
        @test gpu_table.coeffs isa OpenCL.CLArray
    end

    @testset "Pass RGBToSpectrumTable directly to kernel" begin
        n = 16

        # Create test RGB values on GPU
        r_cpu = Float32[0.2, 0.5, 0.8, 1.0, 0.0, 0.3, 0.7, 0.9,
                        0.1, 0.4, 0.6, 0.8, 0.2, 0.5, 0.7, 1.0]
        g_cpu = Float32[0.3, 0.5, 0.2, 0.0, 1.0, 0.6, 0.4, 0.1,
                        0.8, 0.3, 0.5, 0.2, 0.9, 0.4, 0.6, 0.0]
        b_cpu = Float32[0.4, 0.5, 0.1, 0.0, 0.0, 0.9, 0.3, 0.2,
                        0.5, 0.7, 0.4, 0.3, 0.1, 0.6, 0.2, 0.5]

        r_gpu = adapt(backend, r_cpu)
        g_gpu = adapt(backend, g_cpu)
        b_gpu = adapt(backend, b_cpu)

        # Allocate output arrays
        c0_gpu = KA.allocate(backend, Float32, n)
        c1_gpu = KA.allocate(backend, Float32, n)
        c2_gpu = KA.allocate(backend, Float32, n)

        # Launch kernel with the whole table struct
        kernel! = rgb2spec_table_test_kernel!(backend)
        kernel!(c0_gpu, c1_gpu, c2_gpu, gpu_table, r_gpu, g_gpu, b_gpu; ndrange=n)
        KA.synchronize(backend)

        # Get results
        c0_result = Array(c0_gpu)
        c1_result = Array(c1_gpu)
        c2_result = Array(c2_gpu)

        # Compute expected values on CPU
        for i in 1:n
            expected = rgb_to_spectrum(cpu_table, r_cpu[i], g_cpu[i], b_cpu[i])
            @test c0_result[i] ≈ expected.c0 atol=1e-5
            @test c1_result[i] ≈ expected.c1 atol=1e-5
            @test c2_result[i] ≈ expected.c2 atol=1e-5
        end

        # Cleanup
        finalize(r_gpu)
        finalize(g_gpu)
        finalize(b_gpu)
        finalize(c0_gpu)
        finalize(c1_gpu)
        finalize(c2_gpu)
    end

    @testset "Gray values (special case)" begin
        n = 4

        # Gray values: r == g == b
        gray_vals = Float32[0.0, 0.25, 0.5, 1.0]
        r_gpu = adapt(backend, gray_vals)
        g_gpu = adapt(backend, gray_vals)
        b_gpu = adapt(backend, gray_vals)

        c0_gpu = KA.allocate(backend, Float32, n)
        c1_gpu = KA.allocate(backend, Float32, n)
        c2_gpu = KA.allocate(backend, Float32, n)

        kernel! = rgb2spec_table_test_kernel!(backend)
        kernel!(c0_gpu, c1_gpu, c2_gpu, gpu_table, r_gpu, g_gpu, b_gpu; ndrange=n)
        KA.synchronize(backend)

        c0_result = Array(c0_gpu)
        c1_result = Array(c1_gpu)
        c2_result = Array(c2_gpu)

        # Gray case: c0 = 0, c1 = 0, only c2 varies
        for i in 1:n
            expected = rgb_to_spectrum(cpu_table, gray_vals[i], gray_vals[i], gray_vals[i])
            @test c0_result[i] ≈ expected.c0 atol=1e-5
            @test c1_result[i] ≈ expected.c1 atol=1e-5
            @test c2_result[i] ≈ expected.c2 atol=1e-5
        end

        finalize(r_gpu)
        finalize(g_gpu)
        finalize(b_gpu)
        finalize(c0_gpu)
        finalize(c1_gpu)
        finalize(c2_gpu)
    end
end
