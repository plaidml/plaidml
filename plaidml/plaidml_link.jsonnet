local exports = [
  'plaidml_add_composer_dependency',
  'plaidml_add_composer_input',
  'plaidml_add_composer_output',
  'plaidml_add_composer_update',
  'plaidml_add_dimension',
  'plaidml_alloc_applier',
  'plaidml_alloc_buffer',
  'plaidml_alloc_composer',
  'plaidml_alloc_device_enumerator',
  'plaidml_alloc_device_enumerator_with_config',
  'plaidml_alloc_gradient',
  'plaidml_alloc_int64',
  'plaidml_alloc_invoker',
  'plaidml_alloc_invoker_output_shape',
  'plaidml_alloc_placeholder',
  'plaidml_alloc_real',
  'plaidml_alloc_shape',
  'plaidml_alloc_tensor',
  'plaidml_apply_add_dependency',
  'plaidml_apply_add_input',
  'plaidml_apply_alloc_output',
  'plaidml_build_coded_function',
  'plaidml_build_composed_function',
  'plaidml_close_device',
  'plaidml_compute_grad_wrt',
  'plaidml_free_applier',
  'plaidml_free_buffer',
  'plaidml_free_composer',
  'plaidml_free_device_enumerator',
  'plaidml_free_function',
  'plaidml_free_gradient',
  'plaidml_free_invocation',
  'plaidml_free_invoker',
  'plaidml_free_mapping',
  'plaidml_free_shape',
  'plaidml_free_var',
  'plaidml_get_devconf',
  'plaidml_get_devconf_count',
  'plaidml_get_enumerator_config_source',
  'plaidml_get_function_input',
  'plaidml_get_function_input_count',
  'plaidml_get_function_output',
  'plaidml_get_function_output_count',
  'plaidml_get_invalid_devconf',
  'plaidml_get_mapping_base',
  'plaidml_get_mapping_size',
  'plaidml_get_shape_buffer_size',
  'plaidml_get_shape_dimension_count',
  'plaidml_get_shape_dimension_size',
  'plaidml_get_shape_dimension_stride',
  'plaidml_get_shape_element_count',
  'plaidml_get_shape_offset',
  'plaidml_get_shape_type',
  'plaidml_get_version',
  'plaidml_load_function',
  'plaidml_map_buffer_current',
  'plaidml_map_buffer_discard',
  'plaidml_open_device',
  'plaidml_query_devconf',
  'plaidml_save_function',
  'plaidml_save_invoker',
  'plaidml_schedule_invocation',
  'plaidml_set_floatx',
  'plaidml_set_invoker_const',
  'plaidml_set_invoker_input',
  'plaidml_set_invoker_output',
  'plaidml_set_shape_offset',
  'plaidml_shape_set_layout',
  'plaidml_tensor_attach_qparams',
  'plaidml_writeback_mapping',
  'vai_alloc_ctx',
  'vai_cancel_ctx',
  'vai_clear_status',
  'vai_free_ctx',
  'vai_get_perf_counter',
  'vai_internal_set_vlog',
  'vai_last_status',
  'vai_last_status_str',
  'vai_query_feature',
  'vai_set_eventlog',
  'vai_set_logger',
  'vai_set_perf_counter',
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
    VERSION {
      VERS_1.0 {
        /* Export library symbols. */
        global:
    %(exports)s

        /* Hide all other symbols. */
        local: *;
      };
    };
  ||| % { exports: std.lines(['      ' + export + ';' for export in (exports + linux_so_exports)]) },
}
