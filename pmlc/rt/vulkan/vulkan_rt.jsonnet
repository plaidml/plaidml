local exports = [
    // vulkan_rt
    'initVulkan',
    'deinitVulkan',
    'createVulkanLaunchKernelAction',
    'createVulkanMemoryTransferAction',
    'setVulkanLaunchKernelAction',
    'addVulkanLaunchActionToSchedule',
    'submitCommandBuffers',
    'bindBufferFloat32',
    'bindBufferInt64',
    
    // rt
    'plaidml_rt_trace',
    'plaidml_rt_bounds_check',
    'plaidml_rt_xsmm_gemm_f32',
    'print_memref_f32',
];

local linux_so_exports = [
    '__bss_start',
    '_edata',
    '_end',
    '_fini',
    '_init',
];

{
  'vulkan_rt.ld': |||
    %(exports)s
  ||| % { exports: std.lines(['_' + export for export in exports]) },

  'vulkan_rt.def': |||
    LIBRARY PLAIDMLRT
    EXPORTS
    %(exports)s
  ||| % { exports: std.lines(['   ' + export for export in exports]) },

  'vulkan_rt.lds': |||
    VERS_1.0 {
      /* Export library symbols. */
      global:
    %(exports)s

      /* Hide all other symbols. */
      local: *;
    };
  ||| % { exports: std.lines(['      ' + export + ';' for export in (exports + linux_so_exports)]) },
}
