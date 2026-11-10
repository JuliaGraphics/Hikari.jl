module ShadePrecompileProbe
using Hikari, Mantle, Test, Serialization
const H = Hikari
const L = Mantle.Lava
const KA = Hikari.KA

# A device-free description of the opaque triangle/point-light workload.
# The comparison with captured dispatch signatures below pins every argument.
function signatures()
    A(T, N = 1) = L.LavaDeviceArray{T,N}
    C(T) = L.Const{T,1}
    CI = CartesianIndices{1,Tuple{Base.OneTo{Int}}}
    NI = KA.NDIteration
    iteration = NI.NDRange{1,NI.DynamicSize,NI.StaticSize{(32,)},CI,Nothing,Nothing}
    context = KA.CompilerMetadata{NI.DynamicSize,NI.DynamicCheck,Nothing,CI,iteration,Nothing,Nothing}
    ray = H.VPRayWorkItem
    fields = NamedTuple{fieldnames(ray),Tuple{map(A,fieldtypes(ray))...}}
    rays = H.StructArray{ray,1,fields,Int}
    ray_queue = H.WorkQueue{ray,rays,A(Int32)}
    hit_queue = H.WorkQueue{H.VPSurfaceHit,A(H.VPSurfaceHit),A(Int32)}
    inputs = H.ShaderInputs{A(Float32),A(Int32)}
    nodes = H.SurfaceProgramNodes{8,true,UInt32(0x4fe),false}
    queue = H.ClosureQueue{nodes,A(Int32),inputs}
    queues = Base.return_types(H.shade_queues,Tuple{queue})
    @assert length(queues) == 1
    triangle = H.Raycore.Triangle{H.TriangleMeta}
    accel = Mantle.AdaptedAccel{Nothing,A(triangle),A(UInt32),triangle,Nothing,Nothing,A(H.Raycore.InstanceRecord)}
    media = H.Raycore.StaticMultiTypeSet{Tuple{},Tuple{}}
    textures = Tuple{C(A(H.LookupValue)),C(A(H.NodeInstr{H.NodeFeature.CLOSURES}))}
    materials = H.Raycore.SetView{(),(1,3),Tuple{},textures}
    lights = H.Raycore.StaticMultiTypeSet{Tuple{C(H.PointLight{H.RGBSpectrum})},Tuple{}}
    rgb = H.RGBToSpectrumTable{A(Float32),A(Float32,5),A(Float32)}
    tail = (hit_queue,ray_queue,ray_queue,ray_queue,A(Float32),A(Float32),
        accel,C(H.MediumInterfaceIdx),media,materials,lights,rgb,A(H.LightBVHNode),
        A(Int32),A(UInt32),Int32,Int32,Int32,Int32,Bool,Bool,
        H.SobolRNG{A(UInt32)},A(Int32),A(H.PerspectiveCamera),Int32,Int32)
    return [Tuple{typeof(H.gpu_vp_shade_kernel!),context,Q,tail...} for Q in only(queues).parameters]
end

function verify(captured)
    actual = Serialization.deserialize(captured)
    expected = signatures()
    @test length(expected) == length(actual) == 4
    @test expected == actual
    return nothing
end

function prime()
    cache = Mantle.AccessCache()
    for sig in signatures()
        args = Tuple(sig.parameters[2:end])
        interp = L.kernelinterpreter(H.gpu_vp_shade_kernel!,Tuple{args...};workgroup_size=(32,1,1))
        Mantle.accessof(interp,H.gpu_vp_shade_kernel!,args;cache)
    end
    return cache
end
end
