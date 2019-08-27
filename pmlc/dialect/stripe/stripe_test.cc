// Copyright 2019, Intel Corporation

#include <iostream>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/padding_pass.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/stripe/types.h"

#include "tile/codegen/localize.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lib/lib.h"

using namespace vertexai::tile;         // NOLINT
using namespace pmlc::dialect::stripe;  // NOLINT

lang::RunInfo example() {
  using plaidml::edsl::LogicalShape;
  using vertexai::tile::lib::LoadConv2dBnRelu;
  LogicalShape I(PLAIDML_DATA_FLOAT32, {16, 112, 112, 64});
  LogicalShape K(PLAIDML_DATA_FLOAT32, {3, 3, 64, 128});
  LogicalShape C(PLAIDML_DATA_FLOAT32, {128});
  return LoadConv2dBnRelu("foo", I, K, C, {16, 112, 112, 128});
}

int main() {
  printf("Making context + module\n");
  mlir::MLIRContext context;
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  printf("Making a stripe program + fixing locals\n");
  auto prog = lang::GenerateStripe(example());
  codegen::LocalizeBlockPass(codegen::AliasMap(codegen::AliasMap(), prog->entry.get()), prog->entry.get(), {"tmp"});

  printf("Converting to MLIR\n");
  auto func = ToStripeMLIR(&context, *prog);

  printf("Adding function to module\n");
  module.push_back(func);

  printf("Verifying module\n");
  module.verify();

  printf("Doing some passes\n");
  mlir::PassManager pm;
  pm.addPass(mlir::createCSEPass());
  vertexai::tile::codegen::proto::MLIR_PadPass options;
  pm.addPass(new PaddingPass(options));
  if (failed(pm.run(module))) {
    throw std::runtime_error("Invalid goo\n");
  }

  module.verify();

  printf("Dumping modules\n");
  module.dump();

  printf("Converting the other way\n");
  auto prog2 = ToStripe(func);

  printf("New version:\n");
  std::cout << *prog2.entry;
}
