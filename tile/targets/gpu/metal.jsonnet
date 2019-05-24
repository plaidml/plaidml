local PARAMS = {
  metal: {
    CACHE_WIDTH: 64,
    MAX_MEM: [3000],
    SUBGROUP_SIZES: [16],
    GLOBAL_MEM_LAT: 420,
    LOCAL_MEM_LAT: 125,
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
              locate_memory: {
                reqs: ['program'],
                loc: { devs: [{ name: 'GLOBAL', units: [{ offset: 0 }] }] },
              },
            },
            {
              name: 'loc_main',
              locate_memory: {
                reqs: ['main'],
                loc: { devs: [{ name: 'GLOBAL', units: [{ offset: 0 }] }] },
              },
            },

            // Prune indexes
            { name: 'prune_idxs', prune_idxs: { reqs: ['all'] } },

            // Eliminate the dead code first
            { name: 'dead_code_elimination', dead_code_elimination: { reqs: ['all'] } },

            // Lower temps
            { name: 'localize_tmps', localize: { reqs: ['program'], ref_reqs: ['tmp'] } },

            // Reorder Blocks
            //{ "name": "reorder_blocks", "reorder_blocks" : {"reqs": ["program"] } },

            // Padding, disabled for now due ot issues with gradiants
            { name: 'pad', pad: { reqs: ['main'] } },

            // Do subgroup pass
            {
              name: 'subgroup',
              subgroup: {
                reqs: ['contraction'],
                mem_latency: PARAMS[cfg].GLOBAL_MEM_LAT,
                cache_latency: PARAMS[cfg].LOCAL_MEM_LAT,
                max_mem: PARAMS[cfg].MAX_MEM,
                subgroup_sizes: PARAMS[cfg].SUBGROUP_SIZES,
                cache_width: PARAMS[cfg].CACHE_WIDTH,
              },
            },

            // Do a backup codegen on any remaining contractions
            {
              name: 'tile_fallback',
              autotile: {
                reqs: ['contraction'],  // Apply to only dense operations
                outer_set: ['fallback_outer', 'kernel', 'no_threads'],
                inner_set: ['fallback_inner'],
                clear_outer: true,
                acc_idxs: false,
                // With i/o costs zeroed out, and split factor set high, we basically
                // should pick the largest tile that doesn't use accumuation indexes
                input_cost: 0.0,
                output_cost: 0.0,
                split_factor: -100.0,
                only_po2: true,  // Only consider PO2 sizes for speed
              },
            },
            {
              name: 'cache_backup',
              cache: {
                reqs: ['fallback_outer'],
                dirs: ['Out'],
                mem_loc: { devs: [{ name: 'LOCAL' }] },
                xfer_loc: { devs: [{ name: 'DMA' }] },
              },
            },

            // Clean up extra indexes
            // { "name": "subgroup_prune", "prune_idxs": { "reqs": ["main"] } },

            // Next we fuse in any element-wise operations which operate on the output of contraction block
            // We need to fuse through multiple levels of the hierarchy
            { name: 'fuse_contract_eltwise_1', fusion: { a_reqs: ['subgroup_outer'], b_reqs: ['eltwise'] } },
            { name: 'fuse_contract_eltwise_2', fusion: { a_reqs: ['subgroup_thread'], b_reqs: ['eltwise'] } },
            { name: 'fuse_contract_eltwise_3', fusion: { a_reqs: ['subgroup_write'], b_reqs: ['eltwise'] } },

            // Clean things up to allow further optimizations
            { name: 'fuse_clean_1', prune_idxs: { reqs: ['main'] } },
            { name: 'fuse_clean_2', prune_refs: { reqs: ['main'] } },

            // Then we fuse multiple eltwise things
            {
              name: 'fuse_eltwise_eltwise',
              fusion: {
                parent_reqs: ['main'],
                a_reqs: ['eltwise'],
                b_reqs: ['eltwise'],
                output_match: true,
              },
            },

            // Then we 'localize' buffers, which moves any temporaries the only are needed to hold the output
            // of dense computations before they are used on elementwise operations into the interior of the fused blocks
            { name: 'localize_main', localize: { reqs: ['main'] } },
            { name: 'scalarize_main', scalarize: { reqs: ['main'] } },

            // Let's do some additional threading
            {
              name: 'tile_subgroups',
              autotile: {
                reqs: ['subgroup_outer'],  // Apply to only dense operations
                outer_set: ['subgroup_outer', 'kernel'],
                inner_set: ['gpu_thread', 'subgroup_othreads'],
                clear_outer: false,
                only_even: true,
                min_size: 8,
                max_sizes_product: 8,
              },
            },

            // Do a backup codegen on eltwise stuff
            {
              name: 'tile_elt_fallback',
              autotile: {
                reqs: ['eltwise', 'kernel'],  // Apply to eltwise ops
                outer_set: ['fallback_elt_outer', 'kernel', 'no_threads'],
                inner_set: ['fallback_elt_inner'],
                clear_outer: true,
                acc_idxs: false,
                // With i/o costs zeroed out, and split factor set high, we basically
                // should pick the largest tile that doesn't use accumuation indexes
                input_cost: 0.0,
                output_cost: 0.0,
                split_factor: -100.0,
                only_po2: true,  // Only consider PO2 sizes for speed
              },
            },

            // After all fusion, eliminate dead code again
            { name: 'dead_code_elimination', dead_code_elimination: { reqs: ['all'] } },

            { name: 'cleanup1', prune_refs: { reqs: ['main'] } },
            { name: 'cleanup2', prune_idxs: { reqs: ['main'] } },

            { name: 'localize_main', localize: { reqs: ['main'] } },
            { name: 'scalarize_main', scalarize: { reqs: ['main'] } },
          ],
        },
      },
    }
    for cfg in std.objectFields(PARAMS)
  },
}
