using Test, Hikari, Mantle, KernelAbstractions

@kernel unsafe_indices=true inbounds=true function capacity_reservation_test!(items,sizes,@Const(targets),n::Int32,mode::Val)
    j = KernelAbstractions.KernelInterface.get_global_id(Int32).x
    target = j <= n ? targets[j] : Int32(0)
    Hikari.store_capacity_index!(mode,items,sizes,n,j,target)
end

@testset "capacity reservation covers mixed classes and padded subgroups" begin
    backend = Mantle.defaultbackend()
    modes = Hikari.capacity_group_reservation(backend) ? (Val(false),Val(true)) : (Val(false),)
    for n in (0,1,31,33,63,65,127,129), pattern in (:empty,:uniform,:mixed), mode in modes
        targets = Int32[pattern==:empty ? 0 : pattern==:uniform ? 2 : (j%5) for j in 1:n]
        mem = Hikari.DeviceMemory(backend)
        plan = nothing
        try
            items = Hikari.alloc!(mem,Int32,max(4*n,1),-1)
            sizes = Hikari.alloc!(mem,Int32,4)
            inputs = Hikari.alloc!(mem,Int32,max(n,1),0)
            n > 0 && copyto!(inputs,targets)
            g = Mantle.Graph(mem.dev)
            Mantle.dispatch!(g,Hikari.zero_size_counters_kernel!,((sizes,),),1;group=1)
            Mantle.dispatch!(g,capacity_reservation_test!,(items,sizes,inputs,Int32(n),mode),max(n,1);group=128)
            plan = Mantle.record!(Mantle.Plan(g))
            for _ in 1:2
                Mantle.run!(plan)
                Mantle.waitfor!(plan)
                counts,actual = Array(sizes),Array(items)
                @test sum(counts)==count(!iszero,targets)
                for bin in 1:4
                    expected = Int32[j for j in 1:n if targets[j]==bin]
                    @test counts[bin]==length(expected)
                    start = (bin-1)*n
                    @test sort(actual[start+1:start+counts[bin]])==expected
                    @test all(==(-1),actual[start+counts[bin]+1:bin*n])
                end
            end
        finally
            plan===nothing || Mantle.free!(plan)
            Hikari.free!(mem)
        end
    end
end
