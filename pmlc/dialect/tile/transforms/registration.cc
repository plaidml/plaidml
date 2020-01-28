// Copyright 2020, Intel Corporation

// #include "pmlc/dialect/tile/transforms/constant_types.h"
#include "pmlc/dialect/tile/transforms/contraction.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace pmlc::dialect::tile {

static mlir::PassRegistration<ComputeBoundsPass> compute_bounds_pass(  //
    "tile-compute-bounds",                                             //
    "Compute bounds for contractions");

/*

// struct ConstantTypesPass : public OperationPass<ConstantTypesPass>;
struct MyPipelineOptions : public mlir::PassPipelineOptions<mlir::detail::PassOptions::Option<std::string>> {
  // These just forward onto llvm::cl::list and llvm::cl::opt respectively.
  mlir::detail::PassOptions::Option<std::string> floatx_option{*this, "tile-constant-types-floatx",
                                                               llvm::cl::desc("...")};
  mlir::detail::PassOptions::Option<std::string> intx_option{*this, "intx", llvm::cl::desc("...")};
  // mlir::ListOption<int> exampleListOption{*this, "list-flag-name", llvm::cl::desc("...")};
};


 static mlir::PassPipelineRegistration<MyPipelineOptions> constant_types_pass(  //
    "tile-constant-types",                                                     //
    "Set constants to specified types", [](mlir::OpPassManager& pm, const MyPipelineOptions& options) -> void {
      auto pipe_options = options.floatx_option.getArgStr();
      IVLOG(1, "options.floatx_option " << pipe_options.str());
      IVLOG(1, "floatx_option " << options.floatx_option);

      pm.addPass()

      // return std::make_unique<ConstantTypesPass>(DataType::f64,
      // DataType::i64);
    }); */

}  // namespace pmlc::dialect::tile
