#pragma once

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/detail/collective.hpp>
#include <queue>
#include <functional>

#include "graph_util.hpp"
#include "fpp_vertex_permuter.hpp"

inline size_t prep_unterminated (ygm::comm &world, ygm::container::map<uint32_t, VtxInfo>& vertex_map) {
    size_t num_unterminated = 0;

    vertex_map.for_all([&num_unterminated](const uint32_t& vtx, VtxInfo& info){
        if (!info.active) {
            return;
        }

        num_unterminated++;

        if(info.mark_pred && info.mark_desc) 
        {
            info.active = false;
            info.comp_id = info.my_marker;
        } else {
            info.mark_pred = false;
            info.mark_desc = false;
            info.my_marker = -1; 
            info.my_pivot = -1;
            info.wcc_pivot = -1;
        }
    });

    num_unterminated = ygm::sum(num_unterminated, world);

    world.barrier();

    return num_unterminated;
}

inline void shear_edges (ygm::comm &world, ygm::container::map<uint32_t, VtxInfo>& vertex_map) {

    static auto p_vertex_map = &vertex_map;
    size_t num_unterminated = 0; 

    vertex_map.local_for_all([&vertex_map, &num_unterminated](const uint32_t& vtx, VtxInfo& info){

        if (!info.active) {
            return;
        }

        auto check_and_remove_in = [] (const uint32_t& vtx, VtxInfo& info, uint32_t sender, bool s_pred, bool s_desc) {

            auto remove_out = [] (uint32_t, VtxInfo& info, uint64_t edge) {
                info.out.erase(edge);
            };

            if (info.mark_pred != s_pred || info.mark_desc != s_desc) {
                info.in.erase(sender);
                p_vertex_map->async_visit(sender, remove_out, vtx);
            }
        };

        for (auto nbr : info.out) {
            p_vertex_map->async_visit(nbr, check_and_remove_in, vtx, info.mark_pred, info.mark_desc);
        }
    });
    
    world.barrier();
}


inline void prop_pivots (ygm::comm &world, ygm::container::map<uint32_t, VtxInfo>& vertex_map) {
    static auto p_vertex_map = &vertex_map;

    struct comp_pivot_fwd {
        void operator()(const uint32_t& vtx, VtxInfo& info, uint32_t pivot, uint32_t marker){
            if (!info.active || info.mark_desc) {
                return;
            }

            if (pivot == info.wcc_pivot) {
                info.mark_desc = true;
                info.my_marker = marker;

                for (auto nbr : info.out) {
                    p_vertex_map->async_visit(nbr, comp_pivot_fwd(), pivot, marker);
                }
            }
        }
    };

    struct comp_pivot_bwd {
        void operator()(const uint32_t& vtx, VtxInfo& info, uint32_t pivot, uint32_t marker){
            if (!info.active || info.mark_pred) {
                return;
            }

            if (pivot == info.wcc_pivot) {
                info.mark_pred = true;
                info.my_marker = marker;

                for (auto nbr : info.in) {
                    p_vertex_map->async_visit(nbr, comp_pivot_bwd(), pivot, marker);
                }
            }
        }
    };

    vertex_map.local_for_all([&vertex_map](const uint32_t& vtx, VtxInfo& info){

        if (!info.active) {
            return;
        }

        if (info.wcc_pivot == info.my_pivot) {
            info.mark_desc = true;
            info.mark_pred = true;
            info.my_marker = vtx;


            for (auto nbr : info.in) {
                p_vertex_map->async_visit(nbr, comp_pivot_bwd(), info.wcc_pivot, vtx);
            }

            for (auto nbr : info.out) {
                p_vertex_map->async_visit(nbr, comp_pivot_fwd(), info.wcc_pivot, vtx);
            }

        }
    });

    world.barrier();
}


inline void init_wcc_pivots (ygm::comm &world, ygm::container::map<uint32_t, VtxInfo> &vertex_map, size_t iter, uint32_t min, uint32_t max) {
    static auto p_vertex_map = &vertex_map;

    // Need a random seed value to choose who gets to be the pivot
    uint64_t seed = 0x9E3779B97F4A7C15ULL + iter;  // 64-bit golden ratio
    FppPermuter perm(min, max, seed);


    vertex_map.local_for_all([&vertex_map, &perm] (uint32_t vtx, VtxInfo& info) {
        if (info.active) {
            info.my_pivot = perm(vtx);
            info.wcc_pivot = info.my_pivot;
            info.my_marker = vtx;
        }
    });

    world.barrier();
    
    world.cout0() << "Stopped before propagation_start" << std::endl;
    static auto p_world = &world;

    static std::priority_queue<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>, std::greater<std::pair<uint32_t, uint32_t>>> workqueue;

    struct pop_front_and_send {
        void operator() () {

            if (!workqueue.empty()){

                std::pair<uint32_t, uint32_t> front = workqueue.top();
                workqueue.pop();

                p_vertex_map->local_visit(front.second, [](uint32_t vtx, VtxInfo& info, uint32_t queued_pivot){
                    if (queued_pivot != info.wcc_pivot) {
                        return;
                    }


                    auto recv_and_enqueue = [] (uint32_t vtx, VtxInfo& info, uint32_t pivot) {
                        if (!info.active) {
                            return;
                        }

                        if (pivot < info.wcc_pivot) {
                            info.wcc_pivot = pivot;
                            workqueue.push({pivot, vtx});

                            p_world->register_pre_barrier_callback(pop_front_and_send());
                        }
                    };
                

                    for (auto desc : info.out) {
                        p_vertex_map->async_visit(desc, recv_and_enqueue, info.wcc_pivot);
                    }

                    for (auto actr : info.in) {
                        p_vertex_map->async_visit(actr, recv_and_enqueue, info.wcc_pivot);
                    }

                }, front.first);
            }
        }
    };

    vertex_map.local_for_all([&perm](uint32_t vtx, VtxInfo& info){
        if (!info.active) {
            return;
        }

        // preempt unnecessary communication
        for (auto desc : info.out) {
            if (perm(desc) < info.wcc_pivot) {
                return;
            }
        }

        // preempt unnnecessary communication
        for (auto actr : info.in) {
            if (perm(actr) < info.wcc_pivot) {
                return;
            }
        }

        workqueue.push({info.my_pivot, vtx});
        p_world->register_pre_barrier_callback(pop_front_and_send());
    });

    world.barrier();

    YGM_ASSERT_RELEASE(workqueue.size() == 0);
}


inline void trim_trivial (ygm::comm &world, ygm::container::map<uint32_t, VtxInfo>& vertex_map) {
    static auto p_vertex_map = &vertex_map;

    struct trim_vtx {
        void operator()(const uint32_t& vtx, VtxInfo& info, uint32_t sender, bool direction){

            if (!info.active) {
                return;
            }

            // for direction, true = sender had no ancestors, false = no descendants
            if (direction == true) {
                info.in.erase(sender);
            }

            if (direction == false) {
                info.out.erase(sender);
            }

            if (info.in.empty()) {
                info.comp_id = vtx;
                info.active = false;

                for (auto desc : info.out) {
                    p_vertex_map->async_visit(desc, trim_vtx(), vtx, true);
                }

                info.out.clear();
                return;
            }

            if (info.out.empty()) {
                info.comp_id = vtx;
                info.active = false;

                for (auto actr : info.in) {
                    p_vertex_map->async_visit(actr, trim_vtx(), vtx, false);
                }

                info.in.clear();
                return;
            }
        }
    };

    vertex_map.local_for_all([&vertex_map] (uint32_t vtx, VtxInfo& info) {

        if (info.active)
        {
            if(info.in.empty()) {
                info.comp_id = vtx;
                info.active = false;

                for (auto desc : info.out) {
                    p_vertex_map->async_visit(desc, trim_vtx(), vtx, true);
                }
                info.out.clear();
            }

            if(info.out.empty()) {
                info.comp_id = vtx;
                info.active = false;

                for (auto actr : info.in) {
                    p_vertex_map->async_visit(actr, trim_vtx(), vtx, false);
                }
                info.in.clear();
            }
        }
    });

    world.barrier();
}