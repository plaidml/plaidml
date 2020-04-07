local exports = [
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
  'plaidml_rt.ld': |||
    %(exports)s
  ||| % { exports: std.lines(['_' + export for export in exports]) },

  'plaidml_rt.def': |||
    LIBRARY PLAIDMLRT
    EXPORTS
    %(exports)s
  ||| % { exports: std.lines(['   ' + export for export in exports]) },

  'plaidml_rt.lds': |||
    VERS_1.0 {
      /* Export library symbols. */
      global:
    %(exports)s

      /* Hide all other symbols. */
      local: *;
    };
  ||| % { exports: std.lines(['      ' + export + ';' for export in (exports + linux_so_exports)]) },
}
