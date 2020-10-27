local exports = [
  // core/ffi.h
  'plaidml_buffer_adopt',
  'plaidml_buffer_alloc',
  'plaidml_buffer_clone',
  'plaidml_buffer_data',
  'plaidml_buffer_free',
  'plaidml_buffer_shape',
  'plaidml_buffer_size',
  'plaidml_init',
  'plaidml_integers_free',
  'plaidml_kvps_free',
  'plaidml_program_compile',
  'plaidml_program_free',
  'plaidml_program_get_inputs',
  'plaidml_program_get_outputs',
  'plaidml_program_get_passes',
  'plaidml_program_repr',
  'plaidml_program_save',
  'plaidml_settings_get',
  'plaidml_settings_list',
  'plaidml_settings_load',
  'plaidml_settings_save',
  'plaidml_settings_set',
  'plaidml_shape_alloc',
  'plaidml_shape_free',
  'plaidml_shape_get_dtype',
  'plaidml_shape_get_nbytes',
  'plaidml_shape_get_rank',
  'plaidml_shape_get_sizes',
  'plaidml_shape_get_strides',
  'plaidml_shape_repr',
  'plaidml_shapes_free',
  'plaidml_shutdown',
  'plaidml_string_free',
  'plaidml_string_ptr',
  'plaidml_strings_free',
  'plaidml_version',

  // edsl/ffi.h
  'plaidml_build',
  'plaidml_contraction_add_constraint',
  'plaidml_contraction_add_operand',
  'plaidml_contraction_build',
  'plaidml_dim_expr_free',
  'plaidml_dim_expr_int',
  'plaidml_dim_expr_none',
  'plaidml_dim_expr_op',
  'plaidml_dim_expr_repr',
  'plaidml_expr_bind_dims',
  'plaidml_expr_cast',
  'plaidml_expr_clone',
  'plaidml_expr_constant',
  'plaidml_expr_contraction',
  'plaidml_expr_dim',
  'plaidml_expr_element',
  'plaidml_expr_float',
  'plaidml_expr_free',
  'plaidml_expr_get_dim',
  'plaidml_expr_get_dtype',
  'plaidml_expr_get_rank',
  'plaidml_expr_get_shape',
  'plaidml_expr_input',
  'plaidml_expr_int',
  'plaidml_expr_intrinsic',
  'plaidml_expr_pragma',
  'plaidml_expr_ptr',
  'plaidml_expr_repr',
  'plaidml_expr_uint',
  'plaidml_poly_expr_dim',
  'plaidml_poly_expr_free',
  'plaidml_poly_expr_index',
  'plaidml_poly_expr_literal',
  'plaidml_poly_expr_op',
  'plaidml_poly_expr_repr',
  'plaidml_targets_get',
  'plaidml_tuple_free',
  'plaidml_value_clone',
  'plaidml_value_dim',
  'plaidml_value_dim_get',
  'plaidml_value_expr',
  'plaidml_value_expr_get',
  'plaidml_value_float',
  'plaidml_value_float_get',
  'plaidml_value_free',
  'plaidml_value_get_kind',
  'plaidml_value_int',
  'plaidml_value_int_get',
  'plaidml_value_none',
  'plaidml_value_repr',
  'plaidml_value_str',
  'plaidml_value_str_get',
  'plaidml_value_tuple',
  'plaidml_value_tuple_get',

  // op/ffi.h
  'plaidml_op_make',

  // exec/ffi.h
  'plaidml_devices_get',
  'plaidml_exec_init',
  'plaidml_executable_free',
  'plaidml_executable_run',
  'plaidml_jit',
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
