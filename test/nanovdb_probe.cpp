// Optional interoperability check against upstream NanoVDB (v32).
// c++ -std=c++17 -O2 -I<openvdb>/nanovdb nanovdb_probe.cpp -lz -o nanovdb_probe
// ./nanovdb_probe density.nvdb
#define NANOVDB_USE_ZIP
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridValidator.h>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    auto handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(argv[1]);
    auto grid = handle.grid<float>();
    if (!grid) return 3;
    auto world = grid->indexToWorld(nanovdb::Vec3d(0.0));
    std::cout << "name=" << grid->gridName() << " leaves=" << grid->tree().nodeCount(0)
              << " value(0,0,0)=" << grid->tree().getValue(nanovdb::Coord(0,0,0))
              << " value(8,8,8)=" << grid->tree().getValue(nanovdb::Coord(8,8,8))
              << " world(0,0,0)=" << world[0] << "," << world[1] << "," << world[2] << "\n";
    const bool valid = nanovdb::isValid(*grid, true, true);
    std::cout << "valid=" << valid << "\n";
    return valid ? 0 : 1;
}
