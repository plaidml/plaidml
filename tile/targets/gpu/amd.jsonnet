local PARAMS = {
  amd: {
    LOCAL_MEM_KIB: 20,
    NUM_THREADS: 256,
    CACHE_WIDTH: 1024, 
    NUM_UNITS: 32
  },
};

{
  configs: {
    [cfg]: {
      stages: {
        default: {
          // Define the stripe passes
          passes: [
            // First, we place all the initial buffer in global memory (DRAM)
            {
              name: 'loc_program',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass',
                reqs: ['program'],
                loc: { devs: [{ name: 'GLOBAL', units: [{ offset: 0 }] }] },
              },
            },
            {
              name: 'loc_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass',
                reqs: ['main'],
                loc: { devs: [{ name: 'GLOBAL', units: [{ offset: 0 }] }] },
              },
            },

            // Prune indexes
            {
              name: 'prune_idxs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['all'],
              },
            },

            // Eliminate the dead code first
            {
              name: 'dead_code_elimination',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.DeadCodeEliminationPass',
                reqs: ['all'],
              },
            },

            // Lower temps
            {
              name: 'localize_tmps',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocalizePass',
                reqs: ['program'],
                ref_reqs: ['tmp'],
              },
            },

            // Reorder Blocks
            {
              name: 'reorder_blocks',
              pass : {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ReorderBlocksPass',
              }
            },

            // Pad tensors to remove inner conditionals
            {
              name: 'pad',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PadPass',
                reqs: ['main'],
              },
            },

            {
              name: 'tile_contract',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass',
                reqs: ['contraction'],
                outer_set: ['contract_outer', 'kernel'],
                inner_set: ['contract_inner'],
                fail_outer_set: ['contract_unexpanded', 'kernel'],
                fail_inner_set: ['no_threads'],
                clear_outer: true,
                acc_idxs: true,
                only_even: true,
                split_factor: -0.1,
                max_total_size: PARAMS[cfg].LOCAL_MEM_KIB * 1024,
                min_out_size: PARAMS[cfg].NUM_THREADS,
                cache_width: PARAMS[cfg].CACHE_WIDTH,
              }
            },

            {
              name: 'tile_middle',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass',
                reqs: ['contract_outer'],
                inner_set: ['contract_middle'],
                acc_idxs: false,
                input_cost: 0.0, 
                output_cost: 0.0,
                split_factor: -100.0,
                only_even: true,
              }
            },

            {
              name: 'prune_idxs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['all'],
              },
            },

            {
              name: 'cache_input',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.CachePass',
                reqs: ['contract_middle'],
                ref: 'contract_inner',
                dirs: [ 'In' ],
                mem_loc: { 'devs': [{'name': 'LOCAL', 'units': [{'offset': 0}]}] },
                xfer_loc: { 'devs': [{'name': 'DMA', 'units': [{'offset': 0}]}] },
              }
            },

            {
              name: 'fuse_contract_eltwise',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                a_reqs: ['contract_outer'],
                b_reqs: ['eltwise'],
                output_match: true,
              }
            },

            {
              name: 'fuse_eltwise_eltwise',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                parent_reqs: ['main'],
                a_reqs: ['eltwise'],
                b_reqs: ['eltwise'],
                output_match: true,
              } 
            },

            // Then we 'localize' buffers, which moves any temporaries the only are needed to hold the output
            // of dense computations before they are used on elementwise operations into the interior of the fused blocks
            {
              name: 'localize_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocalizePass',
                reqs: ['main'],
              },
            },

            // After all fusion, eliminate dead code again
            {
              name: 'dead_code_elimination',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.DeadCodeEliminationPass',
                reqs: ['all'],
              },
            },

            {
              name: 'cache_output',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.CachePass',
                reqs: ['contract_outer'],
                ref: 'contract_inner',
                dirs: [ 'Out', 'InOut' ],
                mem_loc: { 'devs': [{'name': 'LOCAL', 'units': [{'offset': 0}]}] },
                xfer_loc: { 'devs': [{'name': 'DMA', 'units': [{'offset': 0}]}] },
              } 
            },

            {
              name: 'cache_unexpanded',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.CachePass',
                reqs: ['contract_unexpanded'],
                ref: 'no_threads',
                dirs: [ 'Out' ],
                mem_loc: { 'devs': [{'name': 'LOCAL', 'units': [{'offset': 0}]}] },
                xfer_loc: { 'devs': [{'name': 'DMA', 'units': [{'offset': 0}]}] },
              }
            },

            {
              name: 'prune_idxs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['all'],
              },
            },

            {
              name: 'fuse_inner',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                parent_reqs: ['contract_outer'],
                fused_set: ['cache'],
                exclude: ['contract_middle'],
              } 
            },

            {
              name: 'tile_eltwise',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass',
                reqs: ['eltwise', 'kernel'],
                outer_set: ['eltwise_outer', 'kernel'],
                inner_set: ['eltwise_inner'],
                clear_outer: true,
                only_even: true,
                min_count: PARAMS[cfg].NUM_UNITS,
                min_size: PARAMS[cfg].NUM_THREADS,
                cache_width: PARAMS[cfg].CACHE_WIDTH,
              }
            },

            {
              name: 'localize_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocalizePass',
                reqs: ['main'],
              },
            },

            {
              name: 'scalarize_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ScalarizePass',
                reqs: ['main'],
              },
            },

            ## We now relabel any remaining buffer inside the contractions as local memory
            {
              name: 'make_local',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass',
                reqs: ['contract_outer'],
                loc: { 'devs': [{'name': 'LOCAL', 'units': [{'offset': 0}]}] },
              }
            },

            // After all fusion, eliminate dead code again
            {
              name: 'dead_code_elimination',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.DeadCodeEliminationPass',
                reqs: ['all'],
              },
            },

            {
              name: 'thread_cache',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass',
                reqs: ['cache'],
                outer_set: ['cache_outer', 'gpu_thread'],
                inner_set: ['cache_threads'],
                only_even: true,
                max_sizes_product: PARAMS[cfg].NUM_THREADS,
                cache_width: PARAMS[cfg].CACHE_WIDTH,
                loc_name: 'GLOBAL',
                flip: true,
              }
            },

            {
              name: 'thread_contract',
              pass : {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ThreadInnerPass',
                reqs: ['contract_inner'],
                threads: PARAMS[cfg].NUM_THREADS,
              }
            },

            {
              name: 'thread_eltwise',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass',
                reqs: ['eltwise_inner'],
                outer_set: ['eltwise_struct', 'gpu_thread'],
                inner_set: ['eltwise_threads'],
                only_even: true,
                max_sizes_product: PARAMS[cfg].NUM_THREADS,
                cache_width: PARAMS[cfg].CACHE_WIDTH,
                loc_name: 'GLOBAL',
                flip: true,
              }
            },

            {
              name: 'thread_elemwise',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ThreadInnerPass',
                reqs: ['kernel', 'eltwise'],
                threads: PARAMS[cfg].NUM_THREADS,
              } 
            },

            {
              name: 'loc_gpu',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocateInnerBlockPass',
                reqs: ['kernel'],
                loc: { 'devs': [{'name': 'GPU', 'units': [{'offset': 0}]}] }
              }
            },

            {
              name: 'cleanup1',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneRefinementsPass',
                reqs: ['main'],
              },
            },
            {
              name: 'cleanup2',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['main'],
              },
            },

            {
              name: 'localize_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocalizePass',
                reqs: ['main'],
              },
            },

            {
              name: 'scalarize_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ScalarizePass',
                reqs: ['main'],
              },
            },

            {
              name: 'compute_deps',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ComputeDepsPass', 
              }
            }, 

            {
              name: 'place_program',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.MemoryPlacementPass',
                reqs: ['program'],
                locs: [{ "devs": [{"name": "GLOBAL", "units": [{"offset": 0}]}] }],
                alignment: 4,
              }
            }
          ],
        },
      },
    }
    for cfg in std.objectFields(PARAMS)
  },
}
