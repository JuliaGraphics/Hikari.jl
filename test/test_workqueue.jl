using Test
using Hikari, Lava, Mantle
using Hikari: WorkQueue, should_use_soa, allocate_array, DeviceMemory
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

# `WorkQueue`'s own behaviour: construction, the atomic push, empty!, and what
# happens past capacity.
#
# This coverage existed as `test/workqueue.jl`, which `runtests.jl` never
# referenced in the entire history of the repo — so none of it had ever run. It
# was written against `WorkQueue{T}(backend, capacity)` and a `cleanup!` that no
# longer exist. Ported to the current API (a queue takes a `DeviceMemory`, and
# the pool reclaims it) and actually wired in.
#
# Everything here goes through a KERNEL, because that is the only way anything
# calls `push!`: every call site in `src/integrators/volpath/` is inside a
# `@kernel` body, and there is no host-side caller. Pushing from the host is a
# different lowering — Atomix's pointer atomic over a plain `Vector` instead of
# a device atomic over a `LavaDeviceArray` reached through `Adapt` — so a
# host-backend test would exercise a path production never takes while looking
# like it covered the queue. The original file's CPU section did exactly that,
# which is the likeliest reason it was never wired in.

@kernel function wq_push_kernel!(queue)
    i = @index(Global)
    push!(queue, Int32(i * 10))
end

# Reads back through the queue's own indexing, on the device.
@kernel function wq_read_double_kernel!(output, queue)
    i = @index(Global)
    if i <= queue.capacity
        @inbounds output[i] = queue.items[i] * Int32(2)
    end
end

@testset "WorkQueue" begin
    backend = Mantle.defaultbackend()

    @testset "construction" begin
        mem = DeviceMemory(backend)
        queue = WorkQueue{Int32}(mem, 100)
        @test queue.capacity == 100
        @test length(queue) == 0
        Hikari.free!(mem)
    end

    @testset "kernel push! lands items and advances the count" begin
        mem = DeviceMemory(backend)
        queue = WorkQueue{Int32}(mem, 100)
        wq_push_kernel!(backend)(queue; ndrange=5)
        KA.synchronize(backend)

        @test length(queue) == 5
        # Order is whatever the atomic handed out, so compare as a set.
        @test sort(Array(queue.items)[1:5]) == Int32[10, 20, 30, 40, 50]
        Hikari.free!(mem)
    end

    @testset "a kernel reads back what another kernel pushed" begin
        mem = DeviceMemory(backend)
        queue = WorkQueue{Int32}(mem, 10)
        output = Hikari.alloc!(mem, Int32, 10, Int32(0))

        wq_push_kernel!(backend)(queue; ndrange=5)
        KA.synchronize(backend)
        wq_read_double_kernel!(backend)(output, queue; ndrange=5)
        KA.synchronize(backend)

        @test sort(Array(output)[1:5]) == Int32[20, 40, 60, 80, 100]
        Hikari.free!(mem)
    end

    @testset "empty! resets the count without costing the storage" begin
        mem = DeviceMemory(backend)
        queue = WorkQueue{Int32}(mem, 10)
        wq_push_kernel!(backend)(queue; ndrange=5)
        KA.synchronize(backend)
        @test length(queue) == 5

        empty!(queue)
        @test length(queue) == 0

        # Pushes again after the reset — the storage survived it.
        wq_push_kernel!(backend)(queue; ndrange=3)
        KA.synchronize(backend)
        @test length(queue) == 3
        Hikari.free!(mem)
    end

    # Past capacity the counter keeps climbing while the storage does not: the
    # index is what the atomic returns, and clamping it would hand two threads
    # the same slot. Callers dispatch over `min(length, capacity)` instead.
    @testset "push! past capacity advances the counter, not the storage" begin
        mem = DeviceMemory(backend)
        queue = WorkQueue{Int32}(mem, 5)
        wq_push_kernel!(backend)(queue; ndrange=10)
        KA.synchronize(backend)

        @test length(queue) == 10          # counter reports every attempt
        @test queue.capacity == 5          # capacity is unchanged
        # Exactly the in-capacity slots got written, and no slot was skipped.
        items = Array(queue.items)
        @test count(!iszero, items) == 5
        Hikari.free!(mem)
    end

    @testset "allocate_array falls back to AoS when SoA does not apply" begin
        mem = DeviceMemory(backend)
        @test !should_use_soa(Int32)
        aos = allocate_array(mem, Int32, 10; soa=false)
        @test length(aos) == 10
        # `soa=true` on a type `should_use_soa` rejects still yields a plain array.
        soa = allocate_array(mem, Int32, 10; soa=true)
        @test length(soa) == 10
        @test typeof(soa) == typeof(aos)
        Hikari.free!(mem)
    end

    # The material queues count per slot in one array (`MaterialQueues`), which the round's
    # reset zeroes with the queues' counters. It zeroed the first element of each array.
    @testset "the counter reset zeroes every counter of an array" begin
        mem = DeviceMemory(backend)
        q = WorkQueue{Int32}(mem, 8)
        slots = Hikari.alloc!(mem, Int32, 3, Int32(5))
        wq_push_kernel!(backend)(q; ndrange = 2)
        KA.synchronize(backend)
        @test length(q) == 2
        Hikari.zero_size_counters_kernel!(backend, 1)((q.size, slots); ndrange = 1)
        KA.synchronize(backend)
        @test length(q) == 0
        @test all(iszero, Array(slots))
        Hikari.free!(mem)
    end

    # The queue owns no memory of its own — `free!` on the DeviceMemory is what
    # returns it, and it must return all of it.
    @testset "queues give their memory back to the pool" begin
        mem = DeviceMemory(backend)
        q1 = WorkQueue{Int32}(mem, 64)
        q2 = WorkQueue{Float32}(mem, 64)
        @test q1.capacity == 64
        @test q2.capacity == 64
        @test Hikari.nallocations(mem) == 4        # items + counter, twice
        Hikari.free!(mem)
        @test Hikari.nallocations(mem) == 0
    end
end
