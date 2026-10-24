The node interpreter in `src/textures/nodes/` and the shader materials in `src/materials/` port
code from Blender's Cycles renderer (intern/cycles, Blender v5.2.1). Each ported file names its source files, copyright holders and
SPDX license identifier in its header; the license texts are here:

- `Apache-2.0.txt`: most of the Cycles SVM kernel, and the closures, subsurface walk and lookup
  tables of the shader materials (`shader.jl`, `tables.jl`, `src/materials/shader-material.jl`,
  `src/materials/subsurface.jl`, `src/materials/data/cycles_tables.f32`, from Cycles v5.2.1),
  copyright 2011-2022 Blender Foundation.
- `BSD-3-Clause.txt`: Perlin noise (`perlin.jl`) and the microfacet closures (`shader.jl`,
  `src/materials/shader-material.jl`, from `bsdf_microfacet.h` and `bsdf_util.h`), adapted by
  Cycles from Open Shading Language.
- `MIT.txt`: smooth Voronoi and distance to edge in `voronoi.jl`, after Inigo Quilez.
