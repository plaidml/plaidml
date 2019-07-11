local PARAMS = {
  cpu: {
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
              name: 'stencil_mac',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.StencilPass',
                reqs: ['agg_op_add', 'comb_op_mul'],
                outer_set: ['mac'],
                inner_set: ['mac_inner', 'xsmm'],
                stencils: [
                  {
                    startup_cost: 32,
                    idxs: [
                      { name: 'm', size: 64, outs: [1], ins: [1, 0] },
                      { name: 'n', size: 16, outs: [-1], ins: [0, -1] },
                      { name: 'k', size: 64, outs: [0], ins: [-1, 1] },
                    ],
                  },
                  {
                    startup_cost: 32,
                    idxs: [
                      { name: 'm', size: 64, outs: [1], ins: [0, 1] },
                      { name: 'n', size: 16, outs: [-1], ins: [-1, 0] },
                      { name: 'k', size: 64, outs: [0], ins: [1, -1] },
                    ],
                  },
                ],
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
                clear_outer: true,
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
