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

            // Automatic stencil pass in MLIR
            {  
               name: 'mlir_auto_stencil',
               pass: {
                 '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.MLIR_AutoStencilPass',
                 reqs: ['agg_op_add', 'comb_op_mul'],
                 processors: 56,
                 only_even: [false, false, true],
                 only_po2: [true, true, false],
               },
            },

            {
              name: 'fuse_mac_eltwise',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                a_reqs: ['mac'],
                b_reqs: ['eltwise'],
                fused_set: ['mac'],
              },
            },
            {
              name: 'fuse_mac_inner_eltwise',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                parent_reqs: ['mac'],
                fused_set: ['eltwise'],
                exclude: ['mac_inner'],
                no_inner: true,
              }
            },
            {
              name: 'eltwise fuse',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.FusionPass',
                parent_reqs: ['main'],
                a_reqs: ['eltwise'],
                b_reqs: ['eltwise'],
                fused_set: ['eltwise'],
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
              name: 'prune_idxs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['program'],
              },
            },

            {
              name: 'tile_contract',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass',
                reqs: ['contraction'],
                inner_set: ['contract_inner'],
                outer_set: ['contract_outer', 'kernel', 'cpu_thread'],
                //outer_set: ['contract_outer', 'kernel'],
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
