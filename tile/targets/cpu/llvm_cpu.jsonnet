local PARAMS = {
  llvm_cpu: {
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

            // No-op MLIR pass to test transcoding
            {
               name: 'mlir_nop',
               pass: {
                 '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.MLIR_NopPass',
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
                inner_set: ['mac_inner', 'xsmm', "cpu_thread"],
                is_strict_dims: true,
                stencils: [
                  {
                    startup_cost: 32,
                    idxs: [
                      { name: 'm', size: 64, outs: [1], ins: [0, 1] },
                      { name: 'n', size: -1, outs: [-1], ins: [-1, 0] },
                      { name: 'k', size: 64, outs: [0], ins: [1, -1] },
                    ],
                  },
                  {
                    startup_cost: 32,
                    idxs: [
                      { name: 'm', size: 32, outs: [1], ins: [0, 1] },
                      { name: 'n', size: -1, outs: [-1], ins: [-1, 0] },
                      { name: 'k', size: 32, outs: [0], ins: [1, -1] },
                    ],
                  },
                  {
                    startup_cost: 32,
                    idxs: [
                      { name: 'm', size: 16, outs: [1], ins: [0, 1] },
                      { name: 'n', size: -1, outs: [-1], ins: [-1, 0] },
                      { name: 'k', size: 16, outs: [0], ins: [1, -1] },
                    ],
                  },
                  {
                    startup_cost: 32,
                    idxs: [
                      { name: 'm', size: 48, outs: [1], ins: [0, 1] },
                      { name: 'n', size: -1, outs: [-1], ins: [-1, 0] },
                      { name: 'k', size: 48, outs: [0], ins: [1, -1] },
                    ],
                  },
                  {
                    startup_cost: 32,
                    idxs: [
                      { name: 'm', size: 80, outs: [1], ins: [0, 1] },
                      { name: 'n', size: -1, outs: [-1], ins: [-1, 0] },
                      { name: 'k', size: 80, outs: [0], ins: [1, -1] },
                    ],
                  },
                  {
                    startup_cost: 32,
                    idxs: [
                      { name: 'm', size: 96, outs: [1], ins: [0, 1] },
                      { name: 'n', size: -1, outs: [-1], ins: [-1, 0] },
                      { name: 'k', size: 96, outs: [0], ins: [1, -1] },
                    ],
                  },
                ],
                inputs_set: [{ tags: ['A'] }, { tags: ['B'] }],
                outputs_set: [{ tags: ['C'] }],
              },
            },

            {
              name: 'prune_idxs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['program'],
              },
            },

            {
              name: 'fuse_eltwise_cmp_lt_cond',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                a_reqs: ['eltwise_cmp_lt'],
                b_reqs: ['eltwise_cond'],
                fused_set: ['fuse_eltwise_cond'],
              },
            },

            {
              name: 'eltwise fuse',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                a_reqs: ['eltwise'],
                b_reqs: ['eltwise'],
                fused_set: ['fuse_eltwise'],
              },
            },

            {
              name: 'eltwise_div and eltwise fuse',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                a_reqs: ['eltwise_div'],
                b_reqs: ['eltwise'],
                fused_set: ['fuse_eltwise_div_eltwise'],
              },
            },

            {
              name: 'eltwise and eltwise_div fuse',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                a_reqs: ['eltwise'],
                b_reqs: ['eltwise_div'],
                fused_set: ['fuse_eltwise_div_eltwise'],
              },
            },

            {
              name: 'agg_op_max and eltwise fuse',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                a_reqs: ['agg_op_max'],
                b_reqs: ['eltwise'],
                fused_set: ['fuse_agg_op_max_eltwise'],
                exclude: ['mac'],
              },
            },

            {
              name: 'tile_contract',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass',
                reqs: ['contraction'],
                inner_set: ['contract_inner'],
                outer_set: ['contract_outer', 'kernel', 'cpu_thread'],
                acc_idxs: false,
                input_cost: 0.0, 
                output_cost: 0.0,
                split_factor: -100.0,
                cache_width: PARAMS[cfg].CACHE_WIDTH,
                // Only consider PO2 sizes for speed
                only_po2: true,
              }
            },

            {
              name: 'prune_idxs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['program'],
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
              name: 'dead_code_elimination',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.DeadCodeEliminationPass',
                reqs: ['all'],
              },
            },

            // Locate all the non-user buffers of be in the DRAM arena
            {
              name: 'locate_program',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocateBlocksRefinementsRecursivelyPass',
                reqs: ['program'],
                skip_tags: ['user'],
                loc: { devs: [{ name: 'DRAM' }] },
              },
            },

            // Assign offsets to allocation arena throughout the program.
            {
              name: 'place_program',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.MemoryPlacementPass',
                reqs: ['program'],
                skip_tags: ['user'],
                locs: [{ devs: [{ name: 'DRAM' }] }],
                alignment: 16,
              },
            },

            // Remove unused refinements after fusing, scalarization, and program placement
            {
              name: 'prune_refs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneRefinementsPass',
                reqs: ['program'],
              },
            },

            // Init aggregation outputs
            // Keet this towards the end since other passes are generating intermediate blocks and the initialization 
            // on aggregation transition could break in such cases.
            {
              name: 'init_aggregation_outputs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AggregationBlockOutputInitializationPass',
                reqs: ['program'],
              },
            },
          ],
        },
      },
    }
    for cfg in std.objectFields(PARAMS)
  },
}
