#include "graph_util.hpp"
#include "scc_dcsc_regular.hpp"
#include "fpp_vertex_permuter.hpp"
#include <iostream>

int main(int argc, char **argv)
{
    ygm::comm world(&argc, &argv);

    if (argc != 2) {
        if (world.rank0()) {
            std::cerr << "Usage: " << argv[0] << " <edgelist_file>" << std::endl;
        }
        return 1;
    }

    std::string edgelist_file = argv[1];
    
    ygm::container::map<uint32_t, VtxInfo> result(world);

    create_vertex_map(world, edgelist_file, result);


    world.barrier();
    uint32_t max_vtx = 0;
    uint32_t min_vtx = -1;

    result.for_all([&max_vtx, &min_vtx](const uint32_t& vtx, VtxInfo& info){
        if (vtx > max_vtx) {
            max_vtx = vtx;
        }

        if (vtx < min_vtx) {
            min_vtx = vtx;
        }
    });

    max_vtx = ygm::max(max_vtx, world);
    min_vtx = ygm::min(min_vtx, world);
    world.barrier();


    world.cout0() << "Starting DCSC" << std::endl;

    bool found_all_scc = false;

    size_t iter = 0;
    size_t unterminated = 1;

    while(unterminated) {
        trim_trivial(world, result);
        init_wcc_pivots(world, result, iter, min_vtx, max_vtx);
        prop_pivots(world, result);
        freeze_scc_reset_reached(world, result);
        unterminated = detect_termination(world, result);

        world.cout0() << "Iteration " << iter++ << " left " << unterminated << " unterminated." << std::endl;
    }
    world.barrier();

    uint32_t scc_count = count_sccs(world, result);
    uint32_t largest_scc = count_largest_scc(world, result);

    world.cout0() << "Converged to final SCCs. Enumerated " << scc_count << std::endl;
    world.cout0() << "Largest SCC contains " << largest_scc << std::endl;
}