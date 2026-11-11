using Pkg

# Every job after `setup` loads the depot that job precompiled. A package that is stale here
# (another CPU target, another Julia, a changed source) would be compiled again in each job,
# which is what the shared depot exists to avoid, so name it and stop.
ids = [Base.PkgId(uuid, info.name) for (uuid, info) in Pkg.dependencies()]
stale = sort!([id.name for id in ids if !Base.in_sysimage(id) && !Base.isprecompiled(id)])
isempty(stale) || error("not precompiled for this job: ", join(stale, ", "))
