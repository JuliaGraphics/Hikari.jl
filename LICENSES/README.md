The node texture interpreter in `src/textures/nodes/` ports code from Blender's Cycles renderer
(intern/cycles, Blender v5.2.1). Each ported file names its source files, copyright holders and
SPDX license identifier in its header; the license texts are here:

- `Apache-2.0.txt`: most of the Cycles SVM kernel, and the closures and lookup tables of the shader
  materials (`shader.jl`, `tables.jl`, `src/materials/shader-material.jl`,
  `src/materials/data/cycles_tables.f32`, from Cycles v5.2.1), copyright 2011-2022
  Blender Foundation.
- `BSD-3-Clause.txt`: Perlin noise (`perlin.jl`), adapted by Cycles from Open Shading Language.
- `MIT.txt`: smooth Voronoi and distance to edge in `voronoi.jl`, after Inigo Quilez.
