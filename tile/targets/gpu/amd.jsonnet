local PARAMS = {
  amd_opencl: {
    LOCAL_MEM_KIB: 32,
    NUM_THREADS: 256,
    CACHE_WIDTH: 128,
    NUM_UNITS: 64,
    REGS_MEM_B: 128,
    REG_MEM_LAT: 1,
    LOCAL_MEM_LAT: 30,
    GLOBAL_MEM_LAT: 100,
    ALIGN_SIZE_B: 128,
    MAX_REFS: 1024,
  },
  amd_metal: {
    LOCAL_MEM_KIB: 64,
    NUM_THREADS: 256,
    CACHE_WIDTH: 128,
    NUM_UNITS: 16,
    REGS_MEM_B: 128,
    REG_MEM_LAT: 1,
    LOCAL_MEM_LAT: 30,
    GLOBAL_MEM_LAT: 100,
    ALIGN_SIZE_B: 128,
    MAX_REFS: 31,
  },
};

{
  configs: {
    [cfg]: {
      stages: {
        default: {
          // Define the stripe passes
          passes: [
        
            // Change tags before optimizations
            {
              name: 'kernel_tag',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.KernelTagPass',
                reqs: ['kernel'],
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

            // No-op MLIR pass to induce transcoding
            /*
            {
              name: 'mlir_pad',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.MLIR_PadPass',
                reqs: ['main'],
              },
            },
            */
            
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
                odd_size: true,
                min_out_count: PARAMS[cfg].NUM_UNITS,
                max_total_size: PARAMS[cfg].LOCAL_MEM_KIB * 1024,
                max_sizes_product: PARAMS[cfg].NUM_THREADS * 64,
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
                min_out_count: PARAMS[cfg].NUM_UNITS,
                split_factor: -100.0,
                only_even: true,
              }
            },

            {
              name: 'fuse_eltwise_eltwise',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                parent_reqs: ['main'],
                a_reqs: ['eltwise'],
                b_reqs: ['eltwise'],
                inner_remove_set: ['kernel'],
                output_match: true,
                max_refs: PARAMS[cfg].MAX_REFS,
              }
            },

            {
              name: 'fuse_contract_eltwise',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                a_reqs: ['contract_outer'],
                b_reqs: ['eltwise'],
                inner_remove_set: ['kernel'],
                b_inner_set: ['eltwise_middle'],
                max_refs: PARAMS[cfg].MAX_REFS,
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
                xfer_loc: {},
                odd_size: true,
              }
            },

            {
              name: 'cache_output',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.CachePass',
                reqs: ['contract_outer'],
                ref: 'contract_inner',
                dirs: [ 'Out', 'InOut' ],
                mem_loc: { 'devs': [{'name': 'LOCAL', 'units': [{'offset': 0}]}] },
                xfer_loc: {},
                odd_size: true,
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
              name: 'reduce_constraints',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.IlpConstraintReductionPass',
                reqs: ['all'],
              },
            },

            {
              name: 'fuse_inner',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                parent_reqs: ['contract_outer'],
                fused_set: ['cache', 'eltwise'],
                exclude: ['contract_middle'],
                no_inner: true,
                max_refs: PARAMS[cfg].MAX_REFS,
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
              name: 'tile_eltwise_kernel',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass',
                reqs: ['eltwise', 'kernel'],
                outer_set: ['eltwise_outer', 'kernel'],
                inner_set: ['eltwise_middle'],
                clear_outer: true,
                only_even: true,
                min_count: PARAMS[cfg].NUM_UNITS,
                max_sizes_product: PARAMS[cfg].NUM_THREADS,
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
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ThreadInnerPass',
                reqs: ['cache'],
                outer_set: ['cache_outer', 'gpu_thread'],
                inner_set: ['cache_threads', 'inline'],
                threads: PARAMS[cfg].NUM_THREADS,
              }
            },

            {
              name: 'thread_contract',
              pass : {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ThreadInnerPass',
                reqs: ['contract_inner'],
                outer_set: ['contract_inner', 'gpu_thread'],
                inner_set: ['contract_inner_threads', 'inline'],
                threads: PARAMS[cfg].NUM_THREADS,
              }
            },

            {
              name: 'thread_eltwise',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ThreadInnerPass',
                reqs: ['eltwise_middle'],
                exclude: ['cache'],
                outer_set: ['eltwise_middle', 'gpu_thread'],
                inner_set: ['eltwise_inner', 'inline'],
                threads: PARAMS[cfg].NUM_THREADS,
              }
            },

            // Load cache into registers
            {
              name: 'register_cache_load',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.RegisterCachePass',
                reqs: ['cache_load', 'gpu_thread'],
                dir: 'In',
                local_loc: { 'devs': [{'name': 'LOCAL', 'units': [{'offset': 0}]}] },
                register_loc: { 'devs': [{'name': 'REGISTER', 'units': [{'offset': 0}]}] },
                register_size: PARAMS[cfg].REGS_MEM_B,
                global_memory_latency: PARAMS[cfg].GLOBAL_MEM_LAT,
                local_memory_latency: PARAMS[cfg].LOCAL_MEM_LAT,
                register_latency: PARAMS[cfg].REG_MEM_LAT,
                comp_parent_tag: 'contract_middle',
                index_order: 'cache',
                align_size: PARAMS[cfg].ALIGN_SIZE_B,
              }
            },

            // Store cache into registers
            {
              name: 'register_cache_store',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.RegisterCachePass',
                reqs: ['cache_store', 'gpu_thread'],
                dir: 'Out',
                local_loc: { 'devs': [{'name': 'LOCAL', 'units': [{'offset': 0}]}] },
                register_loc: { 'devs': [{'name': 'REGISTER', 'units': [{'offset': 0}]}] },
                register_size: PARAMS[cfg].REGS_MEM_B,
                global_memory_latency: PARAMS[cfg].GLOBAL_MEM_LAT,
                local_memory_latency: PARAMS[cfg].LOCAL_MEM_LAT,
                register_latency: PARAMS[cfg].REG_MEM_LAT,
                comp_parent_tag: 'contract_middle',
                index_order: 'comp',
                align_size: PARAMS[cfg].ALIGN_SIZE_B,
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
                locs: [{ 'devs': [{'name': 'GLOBAL', 'units': [{'offset': 0}]}] }],
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
