local PARAMS = {
  opencl_cpu: {
    CACHE_WIDTH: 64,
    L1_CACHE_SIZE: 32,
  },
};

{
  configs: {
    [cfg]: {
      stages: {
        default: {
          // Define the stripe passes
          passes: [
            // Lower temps
            {
              name: 'localize_tmps',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocalizePass',
                reqs: ['program'],
                ref_reqs: ['tmp'],
              },
            },

            // Remove unused refinements
            {
              name: 'prune_refs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneRefinementsPass',
                reqs: ['program'],
              },
            },

            {
              name: 'tile_contract',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass',
                // Apply to only dense operations
                reqs: ['contraction'],
                outer_set: ['contract_outer', 'kernel'],
                inner_set: ['contract_inner'],
                // "acc_idxs": false,
                // Only consider PO2 sizes for speed
                only_po2: true,
                // All inputs must fit in local memory
                max_total_size: PARAMS[cfg].L1_CACHE_SIZE * 1024,
                // Since all loads to/from global memory are across a wide bus, use that as the
                // cache_width to optimize for contigous regions of DRAM for each inner block
                cache_width: PARAMS[cfg].CACHE_WIDTH,
              },
            },
          ],
        },
      },
    }
    for cfg in std.objectFields(PARAMS)
  },
}
