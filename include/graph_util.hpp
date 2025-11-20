#pragma once
#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/io/line_parser.hpp>
#include <sstream>
// #include <iostream>

#include <set>

struct VtxInfo {
    std::set<uint32_t> out;
    std::set<uint32_t> in;

    uint64_t comp_id = -1;
    bool active = true;

    uint32_t my_marker = -1;
    uint32_t my_pivot = -1;
    uint32_t wcc_pivot = -1;

    bool mark_pred = false;
    bool mark_desc = false;

    template <class Archive>
    void serialize(Archive& ar) { ar(out, in, comp_id, active, my_pivot, mark_pred, mark_desc); }
};

inline void create_vertex_map(ygm::comm &world, const std::string& edgelist_file, ygm::container::map<uint32_t, VtxInfo>& vertex_map) {
    if (world.rank0()) {
        std::cout << "Reading edges from " << edgelist_file << std::endl;
    }

    static uint64_t edge_count = 0;

    ygm::io::line_parser lp(world, {edgelist_file});
    lp.for_all([&vertex_map](const std::string& line) {
        if (line.empty() || line[0] == '#') {
            return;
        }

        std::istringstream iss(line);
        
        uint32_t src, dst;

        if (iss >> src >> dst) {
            src += 1;
            dst += 1;

            auto add_fwd_edge = [](const uint32_t& src, VtxInfo& info, const uint32_t dst){
                info.out.insert(dst);
                edge_count++;
            };

            auto add_reverse_edge = [](const uint32_t& dst, VtxInfo& info, const uint32_t src){
                info.in.insert(src);
                edge_count++;
            };

            vertex_map.async_visit(src, add_fwd_edge, dst);
            vertex_map.async_visit(dst, add_reverse_edge, src);
        }
    });

    uint64_t local_edge_count = edge_count;

    world.cout() << local_edge_count << ", " << std::endl;

    edge_count = ygm::sum(edge_count, world);

    world.cout0() << "\nNode Count: " << vertex_map.size() << std::endl;
    world.cout0() << "Edge Count: " << edge_count << std::endl;

    world.barrier();
}


inline uint32_t count_sccs( ygm::comm& world, ygm::container::map<uint32_t, VtxInfo>& vertex_map) {
    
    uint32_t local_count = 0;

    vertex_map.local_for_all([&local_count](const uint32_t &vertex, const VtxInfo &info) {
        if (info.comp_id == vertex) {
            local_count++;
        }
    });

    return ygm::sum(local_count, world);
}

inline uint32_t count_largest_scc(ygm::comm& world, ygm::container::map<uint32_t, VtxInfo>& vertex_map) {

    ygm::container::map<uint32_t, uint32_t> scc_sizes(world);

    vertex_map.for_all([&scc_sizes](const uint32_t &vertex, const VtxInfo &info) {
        scc_sizes.async_visit(info.comp_id, [](auto pmap, const uint32_t &scc_id, uint32_t &count) {
            count++;
        });
    });

    uint32_t local_max = 0;

    scc_sizes.for_all([&local_max](const uint32_t &scc_id, const uint32_t &size) {
        local_max = std::max(local_max, size);
    });

    return ygm::max(local_max, world);
}