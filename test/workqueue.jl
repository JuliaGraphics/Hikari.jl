# WorkQueue Tests
#
# Tests for the unified GPU work queue using KernelAbstractions and OpenCL (pocl_jll).

using Test
using Hikari: WorkQueue, cleanup!, should_use_soa, allocate_array, get_array_type
using KernelAbstractions
using KernelAbstractions: @kernel, @index
import KernelAbstractions as KA
using Atomix: @atomic
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
# Test Kernels
# ============================================================================

# Kernel that pushes to queue using the full WorkQueue (via Adapt)
@kernel function push_test_kernel!(queue)
    i = @index(Global)
    push!(queue, Int32(i * 10))
end

# Kernel that reads from queue and doubles the values
@kernel function read_double_kernel!(output, queue)
    i = @index(Global)
    if i <= queue.size[1]
        @inbounds output[i] = queue.items[i] * Int32(2)
    end
end

# ============================================================================
# Tests
# ============================================================================

@testset "WorkQueue" begin
    @testset "CPU Backend" begin
        backend = KA.CPU()

        @testset "Construction and basic operations" begin
            queue = WorkQueue{Int32}(backend, 100)
            @test queue.capacity == 100
            @test length(queue) == 0
            @test isempty(queue)

            cleanup!(queue)
        end

        @testset "push! and length" begin
            queue = WorkQueue{Int32}(backend, 10)

            # Push items (CPU push works directly)
            for i in 1:5
                push!(queue, Int32(i * 10))
            end

            @test length(queue) == 5
            @test !isempty(queue)

            # Check items
            @test queue[1] == 10
            @test queue[2] == 20
            @test queue[5] == 50

            cleanup!(queue)
        end

        @testset "empty!" begin
            queue = WorkQueue{Int32}(backend, 10)

            for i in 1:5
                push!(queue, Int32(i))
            end
            @test length(queue) == 5

            empty!(queue)
            @test length(queue) == 0
            @test isempty(queue)

            cleanup!(queue)
        end

        @testset "setindex!" begin
            queue = WorkQueue{Int32}(backend, 10)

            for i in 1:3
                push!(queue, Int32(0))
            end

            queue[1] = Int32(100)
            queue[2] = Int32(200)

            @test queue[1] == 100
            @test queue[2] == 200

            cleanup!(queue)
        end

        @testset "bounds checking on push!" begin
            queue = WorkQueue{Int32}(backend, 3)

            # Push up to capacity
            idx1 = push!(queue, Int32(1))
            idx2 = push!(queue, Int32(2))
            idx3 = push!(queue, Int32(3))

            @test idx1 == 1
            @test idx2 == 2
            @test idx3 == 3
            @test length(queue) == 3

            # Push beyond capacity - index returned but item not stored
            idx4 = push!(queue, Int32(4))
            @test idx4 == 4  # Index still increments
            @test length(queue) == 4  # Size reports 4

            # But only 3 items are actually stored (capacity)
            @test queue[1] == 1
            @test queue[2] == 2
            @test queue[3] == 3

            cleanup!(queue)
        end
    end

    @testset "OpenCL Backend (pocl)" begin
        backend = get_pocl_backend()

        @testset "Construction" begin
            queue = WorkQueue{Int32}(backend, 100)
            @test queue.capacity == 100
            @test length(queue) == 0

            cleanup!(queue)
        end

        @testset "Kernel push! with WorkQueue (via Adapt)" begin
            queue = WorkQueue{Int32}(backend, 100)

            # Launch kernel that pushes items - passing the whole queue
            kernel! = push_test_kernel!(backend)
            kernel!(queue; ndrange=5)
            KA.synchronize(backend)

            @test length(queue) == 5

            # Check values (order may vary due to atomic operations)
            items = sort(Array(queue.items)[1:5])
            @test items == Int32[10, 20, 30, 40, 50]

            cleanup!(queue)
        end

        @testset "Kernel read after push" begin
            queue = WorkQueue{Int32}(backend, 10)
            output = KA.allocate(backend, Int32, 10)
            KA.fill!(output, Int32(0))

            # Push items
            push_kernel! = push_test_kernel!(backend)
            push_kernel!(queue; ndrange=5)
            KA.synchronize(backend)

            # Read and transform items
            read_kernel! = read_double_kernel!(backend)
            read_kernel!(output, queue; ndrange=5)
            KA.synchronize(backend)

            # Check output (doubled values)
            result = sort(Array(output)[1:5])
            @test result == Int32[20, 40, 60, 80, 100]

            cleanup!(queue)
            finalize(output)
        end

        @testset "empty! on GPU" begin
            queue = WorkQueue{Int32}(backend, 10)

            # Push items via kernel
            kernel! = push_test_kernel!(backend)
            kernel!(queue; ndrange=5)
            KA.synchronize(backend)

            @test length(queue) == 5

            empty!(queue)
            @test length(queue) == 0

            # Can push again after empty
            kernel!(queue; ndrange=3)
            KA.synchronize(backend)

            @test length(queue) == 3

            cleanup!(queue)
        end

        @testset "Overflow protection" begin
            queue = WorkQueue{Int32}(backend, 5)

            # Try to push more than capacity
            kernel! = push_test_kernel!(backend)
            kernel!(queue; ndrange=10)
            KA.synchronize(backend)

            # Size counter goes to 10, but only 5 items stored
            @test length(queue) == 10

            # Only first 5 slots have valid data
            items = Array(queue.items)
            valid_items = sort(filter(x -> x != 0, items[1:5]))
            @test length(valid_items) == 5

            cleanup!(queue)
        end
    end

    @testset "get_array_type" begin
        @testset "CPU" begin
            ArrayType = get_array_type(KA.CPU())
            @test ArrayType == Array
        end

        @testset "OpenCL" begin
            backend = get_pocl_backend()
            ArrayType = get_array_type(backend)
            @test ArrayType == OpenCL.CLArray
        end
    end

    @testset "SOA allocation" begin
        backend = KA.CPU()

        # Simple type - no SOA
        arr_aos = allocate_array(backend, Int32, 10; soa=false)
        @test arr_aos isa Vector{Int32}
        @test length(arr_aos) == 10

        # With soa=true but should_use_soa returns false for Int32
        arr_soa = allocate_array(backend, Int32, 10; soa=true)
        @test arr_soa isa Vector{Int32}  # Falls back to AOS
    end

    @testset "foreach (GPU kernel execution)" begin
        using Adapt

        # Callable struct for foreach testing - counts items
        struct ItemCounter{A}
            count::A
        end
        Adapt.@adapt_structure ItemCounter

        function (c::ItemCounter)(item)
            @atomic c.count[1] += Int32(1)
            return nothing
        end

        # Callable struct that accumulates sum
        struct ItemAccumulator{A}
            sum::A
        end
        Adapt.@adapt_structure ItemAccumulator

        function (a::ItemAccumulator)(item)
            @atomic a.sum[1] += item
            return nothing
        end

        @testset "CPU Backend" begin
            backend = KA.CPU()
            queue = WorkQueue{Int32}(backend, 10)
            counter = zeros(Int32, 1)

            for i in 1:5
                push!(queue, Int32(i * 10))
            end

            cnt = ItemCounter(counter)
            foreach(cnt, backend, queue)

            @test counter[1] == 5

            cleanup!(queue)
        end

        @testset "OpenCL Backend (pocl)" begin
            backend = get_pocl_backend()
            queue = WorkQueue{Int32}(backend, 100)
            counter = KA.allocate(backend, Int32, 1)
            KA.fill!(counter, Int32(0))

            # Push items via kernel
            kernel! = push_test_kernel!(backend)
            kernel!(queue; ndrange=5)
            KA.synchronize(backend)

            # Use foreach to count items
            cnt = ItemCounter(counter)
            foreach(cnt, backend, queue)
            KA.synchronize(backend)

            @test Array(counter)[1] == 5

            cleanup!(queue)
            finalize(counter)
        end

        @testset "foreach with accumulation" begin
            backend = get_pocl_backend()
            queue = WorkQueue{Int32}(backend, 100)
            sum_arr = KA.allocate(backend, Int32, 1)
            KA.fill!(sum_arr, Int32(0))

            # Push items via kernel
            kernel! = push_test_kernel!(backend)
            kernel!(queue; ndrange=5)
            KA.synchronize(backend)

            # Use foreach to sum items
            acc = ItemAccumulator(sum_arr)
            foreach(acc, backend, queue)
            KA.synchronize(backend)

            # Sum of 10 + 20 + 30 + 40 + 50 = 150
            @test Array(sum_arr)[1] == 150

            cleanup!(queue)
            finalize(sum_arr)
        end

        @testset "foreach on empty queue" begin
            backend = get_pocl_backend()
            queue = WorkQueue{Int32}(backend, 10)
            counter = KA.allocate(backend, Int32, 1)
            KA.fill!(counter, Int32(0))

            # Queue is empty - foreach should not launch kernel
            cnt = ItemCounter(counter)
            foreach(cnt, backend, queue)
            KA.synchronize(backend)

            @test Array(counter)[1] == 0

            cleanup!(queue)
            finalize(counter)
        end
    end
end
