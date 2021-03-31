add_subdirectory(intel_gen)
add_subdirectory(intel_gen_ocl_spirv)
add_subdirectory(x86)

pml_cc_library(
  NAME all
  TYPE STATIC
  SRCS all_targets.cc
  DEPS
    pmlc::compiler
    pmlc::target::intel_gen
    pmlc::target::intel_gen_ocl_spirv
    pmlc::target::x86
)

pml_cc_library(
  NAME all_obj
  TYPE OBJECT
  SRCS all_targets.cc
  DEPS
    pmlc::compiler
    pmlc::target::intel_gen
    pmlc::target::intel_gen_ocl_spirv
    pmlc::target::x86
)
