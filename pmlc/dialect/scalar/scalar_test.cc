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

#include "pmlc/dialect/scalar/ops.h"

using namespace mlir;                   // NOLINT
using namespace pmlc::dialect::scalar;  // NOLINT

#define DEBUG_TYPE "pml_scalar"

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

TEST(Scalar, Basic) {
  Location loc = UnknownLoc::get(globalContext());
  auto module = ModuleOp::create(loc);

  auto std_f32 = FloatType::getF32(globalContext());
  auto f32 = ScalarType::get(globalContext(), DataType::FLOAT32);
  std::vector<Type> arg_types{f32};
  std::vector<Type> ret_types{f32};
  auto func_type = FunctionType::get(arg_types, ret_types, globalContext());
  FuncOp func = FuncOp::create(loc, "test", func_type, {});
  func.addEntryBlock();

  OpBuilder builder(func.getBody());
  {
    edsc::ScopedContext scope(builder, loc);
    auto x1 = func.getArgument(0);
    // auto x1 = builder.create<ScalarConstantOp>(loc, f32, builder.getFloatAttr(std_f32, 1.5f));
    auto x2 = builder.create<ScalarConstantOp>(loc, f32, builder.getFloatAttr(std_f32, 1));
    auto x3 = builder.create<AddOp>(loc, f32, x1, x1);
    auto x4 = builder.create<MulOp>(loc, f32, x3, x2);
    builder.create<ReturnOp>(loc, x4.getResult());
  }

  module.push_back(func);
  module.verify();
  module.dump();

  PassManager pm;
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  auto result = pm.run(module);
  EXPECT_FALSE(failed(result));

  module.dump();

  // CallExpr
  //   *add
  //   *div
  //   *mul
  //   neg
  //   *sub
  //   bit_not
  //   *bit_and
  //   *bit_or
  //   *bit_xor
  //   *bit_left
  //   bit_right
  //   *cmp_eq/cmp_ne/cmp_lt/cmp_gt/cmp_le/cmp_ge
  //   abs
  //   sqrt
  //   acos/cos/cosh
  //   asin/sin/sinh
  //   atan/tan/tanh
  //   as_float/as_int/as_uint
  //   exp/log/pow
  //   index
  //   gather/scatter
  //   prng
  //   reshape
  //   shape
  //   *select
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
