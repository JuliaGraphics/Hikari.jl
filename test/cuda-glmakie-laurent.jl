using CUDA, GLMakie
const T=Float32
const V=CuArray{T,2}
CUDA.allowscalar(false)
let
    ls,ns,ms,λ,dt,sr,k,nt,two,eight,fr =1.e-2,800,1.e-4,0.1,2.e-7,0.1,5e7,5000,T(2),T(8),10
    dt2sm=dt^2/ms
    r,r0,r1,r2=1:ns,1:ns-2,2:ns-1,3:ns
    cxs = [T(j-1)*ls for i in r, j in r]
    dx(i,j) = cxs[i,j]-cxs[ns÷2,ns÷2]
    cxc = [cxs[i,j]+sr*(dx(i,j))*exp(-0.5(dx(i,j)^2+dx(j,i)^2)/λ^2)/λ  for i in r, j in r]
    xs,xc,ys,yc=V(cxs),V(cxc),V(cxs'),V(cxc')
    # ys,yc=collect(xs'),collect(xc')
    xp,yp,xt,yt,fx,fy,zs=copy(xc),copy(yc),copy(xc),copy(yc),zero(xc),zero(xc),zero(xc)
    @. zs = sqrt((xc-xs)^2+(yc-ys)^2)
    czs=Array(zs) ; zn=Observable(czs) ; zl=sr*0.05
    f, ax, pl = surface(1:ns, 1:ns, zn, colorrange=(-zl, zl), axis=(; type=Axis3), figure=(; size=(400, 400)))
    display(f)
    Makie.record(f, "output.gif", 1:nt÷fr, framerate=30) do j
        for i in 1:fr
            @views @. fx[r1,r1] = -k*(eight*xc[r1,r1]-xc[r0,r0]-xc[r1,r0]-xc[r2,r0]-xc[r0,r1]-xc[r2,r1]-xc[r0,r2]-xc[r1,r2]-xc[r2,r2])
            @views @. fy[r1,r1] = -k*(eight*yc[r1,r1]-yc[r0,r0]-yc[r1,r0]-yc[r2,r0]-yc[r0,r1]-yc[r2,r1]-yc[r0,r2]-yc[r1,r2]-yc[r2,r2])
            @. xt = two*xc-xp+fx*dt2sm
            @. yt = two*yc-yp+fy*dt2sm
            xc,xp,xt = xt,xc,xp
            yc,yp,yt = yt,yc,yp
        end
        @. zs = sqrt((xc-xs)^2+(yc-ys)^2)
        czs=Array(zs)
        zn[] = czs
        yield()
    end
end
