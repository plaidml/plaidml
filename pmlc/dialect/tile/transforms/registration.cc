// Copyright 2020, Intel Corporation

#include "pmlc/dialect/tile/transforms/constant_types.h"
#include "pmlc/dialect/tile/transforms/contraction.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace pmlc::dialect::tile {

static mlir::PassRegistration<ComputeBoundsPass> compute_bounds_pass(  //
    "tile-compute-bounds",                                             //
    "Compute bounds for contractions");

struct MyPipelineOptions : public mlir::PassPipelineOptions<mlir::detail::PassOptions::Option<int>> {
  // These just forward onto llvm::cl::list and llvm::cl::opt respectively.
  mlir::detail::PassOptions::Option<int> exampleOption{*this, "flag-name", llvm::cl::desc("...")};
  // mlir::ListOption<int> exampleListOption{*this, "list-flag-name", llvm::cl::desc("...")};
};

static mlir::PassPipelineRegistration<MyPipelineOptions> constant_types_pass(  //
    "tile-constant-types",                                                     //
    "Set constants to specified types", [](mlir::OpPassManager& pm, const MyPipelineOptions& options) -> void {
      // IVLOG(1, "options " << options.str());

      // return std::make_unique<ConstantTypesPass>(DataType::f64,
      // DataType::i64);
    });

}  // namespace pmlc::dialect::tile
