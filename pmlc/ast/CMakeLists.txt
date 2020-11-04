pml_cc_library(
  NAME ast
  HDRS
    ast.h
    builder.h
  SRCS
    ast_ops.cc
    ast.cc
    builder.cc
    eval.cc
  DEPS
    pmlc::compiler
    pmlc::dialect::tile::ir
    pmlc::dialect::tile::transforms
    pmlc::util
    LLVMSupport
    MLIRTransforms
)
