// Copyright 2019, Intel Corporation

#include <iostream>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "tile/plaid_ir/dialect.h"
#include "tile/plaid_ir/ops.h"
#include "tile/plaid_ir/padding_pass.h"
#include "tile/plaid_ir/transcode.h"
#include "tile/plaid_ir/types.h"

#include "tile/codegen/localize.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lib/lib.h"

using namespace vertexai::tile;            // NOLINT
using namespace vertexai::tile::plaid_ir;  // NOLINT

lang::RunInfo example() {
  using plaidml::edsl::LogicalShape;
  using vertexai::tile::lib::LoadConv2dBnRelu;
  LogicalShape I(PLAIDML_DATA_FLOAT32, {16, 112, 112, 64});
  LogicalShape K(PLAIDML_DATA_FLOAT32, {3, 3, 64, 128});
  LogicalShape C(PLAIDML_DATA_FLOAT32, {128});
  return LoadConv2dBnRelu("foo", I, K, C, {16, 112, 112, 128});
}

int main() {
  printf("Registering dialect\n");
  mlir::registerDialect<PlaidDialect>();

  printf("Making context + module\n");
  mlir::MLIRContext context;
  auto module = std::make_unique<mlir::Module>(&context);

  printf("Making a stripe program + fixing locals\n");
  auto prog = lang::GenerateStripe(example());
  codegen::LocalizeBlockPass(codegen::AliasMap(codegen::AliasMap(), prog->entry.get()), prog->entry.get(), {"tmp"});

  printf("Converting to MLIR\n");
  auto func = StripeToPlaidIR(&context, *prog);

  printf("Adding function to module\n");
  module->getFunctions().push_back(func);

  printf("Verifying module\n");
  module->verify();

  printf("Doing some passes\n");
  mlir::PassManager pm;
  pm.addPass(mlir::createCSEPass());
  pm.addPass(new PaddingPass());
  if (failed(pm.run(module.get()))) {
    throw std::runtime_error("Invalid goo\n");
  }

  printf("Dumping modules\n");
  module->dump();
}
