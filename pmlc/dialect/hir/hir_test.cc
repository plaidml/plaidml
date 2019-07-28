// Copyright 2019, Intel Corporation

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/Passes.h"

#include "base/util/logging.h"
#include "pmlc/dialect/hir/ops.h"

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::hir;  // NOLINT

#define DEBUG_TYPE "pml_hir"

class Environment : public ::testing::Environment {
  void SetUp() override {  //
    llvm::DebugFlag = true;
  }
};

[[gnu::unused]] auto init = []() {  //
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

static MLIRContext* globalContext() {
  static thread_local MLIRContext context;
  return &context;
}

namespace llvm {

template <>
struct format_provider<Operation> {
  static void format(const Operation& op, raw_ostream& os, StringRef style) {  //
    const_cast<Operation&>(op).print(os);
  }
};

}  // namespace llvm

// struct LoweringPass : public ModulePass<LoweringPass> {
//   void runOnModule() override { IVLOG(1, "LoweringPass"); }
// };

TEST(HIR, Basic) {
  Location loc = UnknownLoc::get(globalContext());
  auto module = ModuleOp::create(loc);

  auto f32 = FloatType::getF32(globalContext());
  std::vector<Type> arg_types;
  std::vector<Type> ret_types{f32};
  auto func_type = FunctionType::get(arg_types, ret_types, globalContext());
  FuncOp func = FuncOp::create(loc, "test", func_type, {});
  func.addEntryBlock();

  OpBuilder builder(func.getBody());
  {
    edsc::ScopedContext scope(builder, loc);
    auto f1 = builder.create<ConstantFloatOp>(loc, llvm::APFloat(7.0f), f32);
    // auto f2 = builder.create<ConstantFloatOp>(loc, llvm::APFloat(3.14f), f32);
    // auto cion = builder.create<ContractionOp>(loc);
    // auto ret = builder.create<AddFOp>(loc, f1, f2);
    // builder.create<ReturnOp>(loc, ret.getResult());
    builder.create<ReturnOp>(loc, f1.getResult());
  }

  module.push_back(func);
  module.verify();

  PassManager pm;
  pm.addPass(createCSEPass());
  // pm.addPass(createCanonicalizerPass());

  auto result = pm.run(module);
  EXPECT_FALSE(failed(result));

  module.dump();

  // CallExpr
  // ContractionExpr
  //   agg_op:
  //     sum
  //     prod
  //     max
  //     min
  //     assign
  //   combo_op:
  //     add
  //     mul
  //     eq
  //     cond
  // DimExprExpr
  // FloatConst
  // IntConst
  // ParamExpr
}
