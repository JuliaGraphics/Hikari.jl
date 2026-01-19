# Minimal failing kernel for GPU SPIR-V debugging
# 4 lines works, 5 lines fails with "Selection must be structured"

using pocl_jll, OpenCL
using KernelAbstractions


# This kernel WORKS (4 lines of bit manipulation)
@kernel function rev4_works!(output)
    i = @index(Global)
    n = UInt32(i)
    n = (n << 16) | (n >> 16)
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8)
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4)
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2)
    @inbounds output[i] = Float32(n)
end

# This kernel FAILS (5 lines of bit manipulation)
@kernel function rev5_fails!(output)
    i = @index(Global)
    n = UInt32(i)
    n = (n << UInt32(16)) | (n >> UInt32(16))
    n = ((n & 0x00ff00ff) << UInt32(8)) | ((n & 0xff00ff00) >> UInt32(8))
    n = ((n & 0x0f0f0f0f) << UInt32(4)) | ((n & 0xf0f0f0f0) >> UInt32(4))
    n = ((n & 0x33333333) << UInt32(2)) | ((n & 0xcccccccc) >> UInt32(2))
    n2 = ((n & 0x55555555) << UInt32(1)) | ((n & 0xaaaaaaaa) >> UInt32(1))  # This 5th line breaks it
    @inbounds output[i] = Float32(n2)
end
@kernel function rev_mulmask!(output)
    i = @index(Global)
    n = UInt32(i)

    n = (n << UInt32(16)) | (n >> UInt32(16))
    n = ((n & 0x00ff00ff) << UInt32(8)) | ((n & 0xff00ff00) >> UInt32(8))
    n = ((n & 0x0f0f0f0f) << UInt32(4)) | ((n & 0xf0f0f0f0) >> UInt32(4))
    n = ((n & 0x33333333) << UInt32(2)) | ((n & 0xcccccccc) >> UInt32(2))

    # final step without shift-by-1
    n = ((n & UInt32(0x55555555)) * UInt32(2)) |
        ((n & UInt32(0xaaaaaaaa)) >> UInt32(1))

    @inbounds output[i] = Float32(n)
end
# Full bitreverse using Base.bitreverse - should use OpenCL builtin bit_reverse
@kernel function rev_base_bitreverse!(output)
    i = @index(Global)
    n = UInt32(i)
    n = bitreverse(n)
    @inbounds output[i] = Float32(n)
end

# Full 5-line manual bitreverse
@kernel function rev_full_manual!(output)
    i = @index(Global)
    n = UInt32(i)
    n = (n << 16) | (n >> 16)
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8)
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4)
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2)
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1)
    @inbounds output[i] = Float32(n)
end


# Test
function test_kernels()
    n = 8
    expected = Float32.(bitreverse.(UInt32.(1:n)))
    println("Expected bitreverse results: ", expected)

    output_gpu = OpenCL.CLArray{Float32}(undef, n)
    backend = KernelAbstractions.get_backend(output_gpu)

    println("\n1. Testing 4-line kernel (partial bitreverse)...")
    kernel4 = rev4_works!(backend, 64)
    kernel4(output_gpu, ndrange=n)
    KernelAbstractions.synchronize(backend)
    println("   Result: ", Array(output_gpu))

    println("\n2. Testing Base.bitreverse (OpenCL builtin)...")
    try
        kernel_base = rev_base_bitreverse!(backend, 64)
        kernel_base(output_gpu, ndrange=n)
        KernelAbstractions.synchronize(backend)
        result = Array(output_gpu)
        println("   Result: ", result)
        println("   Match:  ", all(result .≈ expected))
    catch e
        println("   FAILED: ", e)
    end

    println("\n3. Testing full manual bitreverse...")
    try
        kernel_manual = rev_full_manual!(backend, 64)
        kernel_manual(output_gpu, ndrange=n)
        KernelAbstractions.synchronize(backend)
        result = Array(output_gpu)
        println("   Result: ", result)
        println("   Match:  ", all(result .≈ expected))
    catch e
        println("   FAILED: ", e)
    end
end

# Run if executed directly
test_kernels()
