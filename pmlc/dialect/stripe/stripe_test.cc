// Copyright 2019, Intel Corporation

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/padding_pass.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/stripe/types.h"

#include "base/util/logging.h"
#include "testing/matchers.h"
#include "tile/codegen/compile_pass.h"
#include "tile/codegen/localize.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/runinfo.h"
#include "tile/lib/lib.h"

using namespace vertexai::tile;         // NOLINT
using namespace pmlc::dialect::stripe;  // NOLINT

using ::testing::LinesEq;

class Environment : public ::testing::Environment {
  void SetUp() override {
    plaidml::init();
    plaidml::edsl::init();
  }
};

[[gnu::unused]] auto init = []() {  //
  ::testing::AddGlobalTestEnvironment(new Environment);
  return 0;
}();

lang::RunInfo example() {
  using plaidml::edsl::LogicalShape;
  using vertexai::tile::lib::LoadConv2dBnRelu;
  LogicalShape I(PLAIDML_DATA_FLOAT32, {16, 112, 112, 64});
  LogicalShape K(PLAIDML_DATA_FLOAT32, {3, 3, 64, 128});
  LogicalShape C(PLAIDML_DATA_FLOAT32, {128});
  return LoadConv2dBnRelu("foo", I, K, C, {16, 112, 112, 128});
}

template <typename Pass, typename Config>
std::unique_ptr<mlir::FunctionPassBase> CreatePass(Config config) {
  return std::make_unique<Pass>(config);
}

TEST(Stripe, Transcode) {
  IVLOG(1, "Making context + module");
  mlir::MLIRContext context;

  IVLOG(1, "Making a stripe program + fixing locals");
  auto prog = lang::GenerateStripe(example());
  codegen::LocalizeBlockPass(codegen::AliasMap(codegen::AliasMap(), prog->entry.get()), prog->entry.get(), {"tmp"});

  codegen::CompilerState cstate{prog};

  IVLOG(1, "Adding a memory location");
  codegen::proto::LocateMemoryPass lmp;
  auto lmp_dev = lmp.mutable_loc()->add_devs();
  lmp_dev->set_name("OuterMem");
  lmp_dev->add_units()->set_offset(0);
  lmp_dev = lmp.mutable_loc()->add_devs();
  lmp_dev->set_name("InnerMem");
  lmp_dev->add_units()->set_offset(1);
  codegen::LocateMemoryPass{lmp}.Apply(&cstate);

  IVLOG(1, "Adding an executor location");
  codegen::proto::LocateBlockPass lbp;
  lbp.add_reqs("main");
  auto lbp_dev = lbp.mutable_loc()->add_devs();
  lbp_dev->set_name("OuterExecutor");
  lbp_dev->add_units()->set_offset(0);
  lbp_dev = lbp.mutable_loc()->add_devs();
  lbp_dev->set_name("InnerExecutor");
  lbp_dev->add_units()->set_offset(1);
  codegen::LocateBlockPass{lbp}.Apply(&cstate);

  IVLOG(2, "Original version:");
  IVLOG(2, *prog->entry);

  IVLOG(1, "Converting to MLIR");
  auto module = IntoMLIR(&context, *prog);

  IVLOG(1, "Verifying module");
  if (failed(module->verify())) {
    module->dump();
    throw std::runtime_error("IntoMLIR verification failure");
  }

  IVLOG(1, "Doing some passes");
  mlir::PassManager pm(true);
  pm.addPass(mlir::createCSEPass());
  codegen::proto::MLIR_PadPass options;
  pm.addPass(CreatePass<PaddingPass>(options));
  if (failed(pm.run(*module))) {
    module->dump();
    throw std::runtime_error("MLIR passes failure");
  }

  IVLOG(2, "Dumping module");
  auto moduleOp = *module;
  IVLOG(2, mlir::debugString(moduleOp));

  IVLOG(1, "Converting the other way");
  auto prog2 = FromMLIR(*module);

  IVLOG(2, "New version:");
  IVLOG(2, *prog2->entry);

  auto expected = R"(0: #program #total_macs=14797504512 
block []:1 ( // foo
    #user none new@0x00000000 B<OuterMem[0]/InnerMem[1]>[0] fp32:I(128):(1):512 B
    #user none new@0x00000000 I<OuterMem[0]/InnerMem[1]>[0, 0, 0, 0] fp32:I(16, 112, 112, 64):(802816, 7168, 64, 1):50176 KiB
    #user none new@0x00000000 K<OuterMem[0]/InnerMem[1]>[0, 0, 0, 0] fp32:I(3, 3, 64, 128):(24576, 8192, 128, 1):288 KiB
    #user none new@0x00000000 S<OuterMem[0]/InnerMem[1]>[0] fp32:I(128):(1):512 B
    #user none new@0x00000000 _X2<OuterMem[0]/InnerMem[1]>[0, 0, 0, 0] fp32:I(16, 112, 112, 128):(1605632, 14336, 128, 1):100352 KiB
) {
  0: #main 
  block<OuterExecutor[0]/InnerExecutor[1]> []:1 ( // main
      in B<OuterMem[0]/InnerMem[1]>[0] fp32:I(128):(1):512 B, E(128):512 B
      in I<OuterMem[0]/InnerMem[1]>[0, 0, 0, 0] fp32:I(16, 112, 112, 64):(802816, 7168, 64, 1):50176 KiB, E(16, 112, 112, 64):50176 KiB
      in K<OuterMem[0]/InnerMem[1]>[0, 0, 0, 0] fp32:I(3, 3, 64, 128):(24576, 8192, 128, 1):288 KiB, E(3, 3, 64, 128):288 KiB
      none new@0x00000000 O[0, 0, 0, 0] fp32:I(16, 112, 112, 128):(1605632, 14336, 128, 1):100352 KiB
      in S<OuterMem[0]/InnerMem[1]>[0] fp32:I(128):(1):512 B, E(128):512 B
      none new@0x00000000 _X0[0, 0, 0, 0] fp32:I(16, 112, 112, 128):(1605632, 14336, 128, 1):100352 KiB
      none new@0x00000000 _X1[0, 0, 0, 0] fp32:I(16, 112, 112, 128):(1605632, 14336, 128, 1):100352 KiB
      out _X2<OuterMem[0]/InnerMem[1]>[0, 0, 0, 0] fp32:I(16, 112, 112, 128):(1605632, 14336, 128, 1):100352 KiB, E(16, 112, 112, 128):100352 KiB
  ) {
    0: #agg_op_add #comb_op_mul #contraction #kernel 
    block [ci:64, co:128, k0:3, k1:3, n:16, x0:112, x1:112]:14797504512 ( // kernel_0(I,K)
        // O[n, x0, x1, co : 16, 112, 112, 128] = +(I[n, -1 + k0 + x0, -1 + k1 + x1, ci] * K[k0, k1, ci, co])
        112 - k1 - x1 >= 0
        -1 + k1 + x1 >= 0
        112 - k0 - x0 >= 0
        -1 + k0 + x0 >= 0
        #contraction in I<OuterMem[0]/InnerMem[1]>[n, -1 + k0 + x0, -1 + k1 + x1, ci] fp32:I(1, 1, 1, 1):(802816, 7168, 64, 1):4 B, E(16, 114, 114, 64):51984 KiB
        #contraction in K<OuterMem[0]/InnerMem[1]>[k0, k1, ci, co] fp32:I(1, 1, 1, 1):(24576, 8192, 128, 1):4 B, E(3, 3, 64, 128):288 KiB
        out O[n, x0, x1, co] fp32:I(1, 1, 1, 1):(1605632, 14336, 128, 1):4 B, E(16, 112, 112, 128):100352 KiB
    ) {
      0: $s0 = load(I)
      1: $s1 = load(K)
      2: $s2 = mul($s0, $s1)
      3: O = store($s2)
    }
    1: #eltwise #eltwise_add #kernel 
    block [i1:16, i2:112, i3:112, i4:128]:25690112 ( // kernel_1(O,B)
        // _X0 = add(O, B)
        #eltwise_add in B<OuterMem[0]/InnerMem[1]>[i4] fp32:I(1):(1):4 B, E(128):512 B
        #eltwise_add in O[i1, i2, i3, i4] fp32:I(1, 1, 1, 1):(1605632, 14336, 128, 1):4 B, E(16, 112, 112, 128):100352 KiB
        out _X0[i1, i2, i3, i4] fp32:I(1, 1, 1, 1):(1605632, 14336, 128, 1):4 B, E(16, 112, 112, 128):100352 KiB
    ) {
      0: $s3 = load(O)
      1: $s4 = load(B)
      2: $s5 = add($s3, $s4)
      3: _X0 = store($s5)
    }
    2: #eltwise #eltwise_mul #kernel 
    block [i1:16, i2:112, i3:112, i4:128]:25690112 ( // kernel_2(_X0,S)
        // _X1 = mul(_X0, S)
        #eltwise_mul in S<OuterMem[0]/InnerMem[1]>[i4] fp32:I(1):(1):4 B, E(128):512 B
        #eltwise_mul in _X0[i1, i2, i3, i4] fp32:I(1, 1, 1, 1):(1605632, 14336, 128, 1):4 B, E(16, 112, 112, 128):100352 KiB
        out _X1[i1, i2, i3, i4] fp32:I(1, 1, 1, 1):(1605632, 14336, 128, 1):4 B, E(16, 112, 112, 128):100352 KiB
    ) {
      0: $s6 = load(_X0)
      1: $s7 = load(S)
      2: $s8 = mul($s6, $s7)
      3: _X1 = store($s8)
    }
    3: #eltwise #eltwise_relu #kernel 
    block [i1:16, i2:112, i3:112, i4:128]:25690112 ( // kernel_3(_X1)
        // _X2 = relu(_X1)
        #eltwise_relu in _X1[i1, i2, i3, i4] fp32:I(1, 1, 1, 1):(1605632, 14336, 128, 1):4 B, E(16, 112, 112, 128):100352 KiB
        out _X2<OuterMem[0]/InnerMem[1]>[i1, i2, i3, i4] fp32:I(1, 1, 1, 1):(1605632, 14336, 128, 1):4 B, E(16, 112, 112, 128):100352 KiB
    ) {
      0: $s9 = load(_X1)
      1: $s10 = relu($s9)
      2: _X2 = store($s10)
    }
  }
}
)";

  EXPECT_THAT(to_string(*prog2->entry), LinesEq(expected));
}
