local exports = [
  // core/ffi.h
  'plaidml_string_ptr',
  'plaidml_string_free',
  'plaidml_init',
  'plaidml_shutdown',
  'plaidml_version',
  'plaidml_settings_list_count',
  'plaidml_settings_list',
  'plaidml_settings_get',
  'plaidml_settings_set',
  'plaidml_settings_load',
  'plaidml_settings_save',
  'plaidml_shape_free',
  'plaidml_shape_alloc',
  'plaidml_shape_repr',
  'plaidml_shape_get_ndims',
  'plaidml_shape_get_dtype',
  'plaidml_shape_get_dim_size',
  'plaidml_shape_get_dim_stride',
  'plaidml_shape_get_nbytes',
  'plaidml_buffer_free',
  'plaidml_buffer_alloc',
  'plaidml_buffer_mmap_current',
  'plaidml_buffer_mmap_discard',
  'plaidml_view_free',
  'plaidml_view_data',
  'plaidml_view_size',
  'plaidml_view_writeback',

  // edsl/ffi.h
  'plaidml_edsl_init',
  'plaidml_logical_shape_alloc',
  'plaidml_logical_shape_free',
  'plaidml_logical_shape_into_tensor_shape',
  'plaidml_logical_shape_repr',
  'plaidml_logical_shape_get_ndims',
  'plaidml_logical_shape_get_dtype',
  'plaidml_logical_shape_get_dim_int',
  'plaidml_poly_expr_free',
  'plaidml_poly_expr_repr',
  'plaidml_poly_expr_dim',
  'plaidml_poly_expr_index',
  'plaidml_poly_expr_literal',
  'plaidml_poly_expr_op',
  'plaidml_dim_expr_free',
  'plaidml_dim_expr_repr',
  'plaidml_dim_expr_none',
  'plaidml_dim_expr_int',
  'plaidml_dim_expr_get_int',
  'plaidml_dim_expr_op',
  'plaidml_expr_free',
  'plaidml_expr_get_shape',
  'plaidml_expr_bind_shape',
  'plaidml_expr_bind_dims',
  'plaidml_expr_get_kind',
  'plaidml_expr_repr',
  'plaidml_expr_clone',
  'plaidml_expr_get_dim',
  'plaidml_expr_dim',
  'plaidml_expr_placeholder',
  'plaidml_expr_param_reset',
  'plaidml_expr_none',
  'plaidml_expr_tuple',
  'plaidml_expr_tuple_get_count',
  'plaidml_expr_tuple_get_exprs',
  'plaidml_expr_str',
  'plaidml_expr_str_get_value',
  'plaidml_expr_int',
  'plaidml_expr_int_get_value',
  'plaidml_expr_float',
  'plaidml_expr_float_get_value',
  'plaidml_expr_cast',
  'plaidml_expr_call',
  'plaidml_expr_index_map',
  'plaidml_expr_size_map',
  'plaidml_expr_contraction',
  'plaidml_expr_contraction_add_constraint',
  'plaidml_expr_contraction_set_no_reduce',
  'plaidml_expr_contraction_set_use_default',
  'plaidml_expr_gradient',
  'plaidml_deriv_register',
  'plaidml_program_free',
  'plaidml_program_evaluate',
  'plaidml_program_repr',
  'plaidml_program_runinfo',

  // op/ffi.h
  'plaidml_op_init',
  'plaidml_op_make',

  // exec/ffi.h
  'plaidml_exec_init',
  'plaidml_device_list_count',
  'plaidml_device_list',
  'plaidml_target_list_count',
  'plaidml_target_list',
  'plaidml_compile',
  'plaidml_executable_free',
  'plaidml_executable_run',
];

local linux_so_exports = [
  '__bss_start',
  '_edata',
  '_end',
  '_fini',
  '_init',
];

{
  'plaidml.ld': |||
    %(exports)s
  ||| % { exports: std.lines(['_' + export for export in exports]) },

  'plaidml.def': |||
    LIBRARY PLAIDML
    EXPORTS
    %(exports)s
  ||| % { exports: std.lines(['   ' + export for export in exports]) },

  'plaidml.lds': |||
    VERS_1.0 {
      /* Export library symbols. */
      global:
    %(exports)s

      /* Hide all other symbols. */
      local: *;
    };
  ||| % { exports: std.lines(['      ' + export + ';' for export in (exports + linux_so_exports)]) },
}
