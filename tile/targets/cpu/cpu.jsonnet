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

            // Assign offsets to allocation arena throughout the program.
            {
              name: 'place_program',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.MemoryPlacementPass',
                reqs: ['program'],
                locs: [{ devs: [{ name: 'DRAM', units: [{ offset: 0 }] }] }],
                alignment: 4,
              },
            },

            // Pad tensors to remove inner conditionals
            {
              name: 'pad',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PadPass',
                reqs: ['main'],
              },
            },

            // Stencil pass to tile the data for XSMM
            {
              name: 'stencil_mac',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.StencilPass',
                reqs: ['agg_op_add', 'comb_op_mul'],
                outer_set: ['mac'],
                inner_set: ['mac_inner', 'xsmm'],
                is_strict_dims: true,
                stencils: [
                  {
                    startup_cost: 32,
                    idxs: [
                      { name: 'm', size: i, outs: [-1], ins: [-1, 0] },
                      { name: 'n', size: j, outs: [1], ins: [0, 1] },
                      { name: 'k', size: k, outs: [0], ins: [1, -1] },
                    ],
                  } for i in [8, 16, 32, 48, 64, 80, 96] for j in [8, 16, 32, 64, 48, 80, 96] for k in [8, 16, 32, 48, 64, 80, 96]
                ],
                inputs_set: [{tags: ["A"]}, {tags: ["B"]}],
                outputs_set: [{tags: ["C"]}],
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
