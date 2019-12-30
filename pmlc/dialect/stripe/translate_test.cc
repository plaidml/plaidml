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
#include "plaidml2/edsl/helper.h"
#include "plaidml2/op/op.h"
#include "testing/matchers.h"
#include "tile/codegen/compile_pass.h"
#include "tile/codegen/localize.h"

using namespace plaidml::edsl;          // NOLINT
using namespace pmlc::dialect::stripe;  // NOLINT
using namespace vertexai::tile;         // NOLINT

using ::testing::LinesEq;

template <typename Pass, typename Config>
std::unique_ptr<mlir::Pass> CreatePass(Config config) {
  return std::make_unique<Pass>(config);
}

// Stripe Classic <-> Stripe MLIR transcoding tests are parameterized by whether
// they should add location info or not, since there've been some subtle
// transcoding issues when location-adding top-level refinements are or aren't
// in place.
class TranslateTest : public ::testing::TestWithParam<bool> {};

static void RunTest(const Program& program, bool addLocations) {
  IVLOG(1, "Making context + module");
  mlir::MLIRContext context;

  IVLOG(1, "Making a stripe program + fixing locals");
  auto stripe = plaidml::edsl::ConvertIntoStripe(program);

  if (addLocations) {
    codegen::CompilerState cstate{stripe};

    IVLOG(1, "Adding a memory location");
    codegen::proto::LocateMemoryPass lmp;
    auto lmp_dev = lmp.mutable_loc()->add_devs();
    lmp_dev->set_name("OuterMem");
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
  }

  IVLOG(2, "Original version:");
  IVLOG(2, *stripe->entry);

  IVLOG(1, "Converting to MLIR");
  auto module = IntoMLIR(&context, *stripe);

  IVLOG(1, "Verifying module");
  if (failed(module->verify())) {
    module->dump();
    throw std::runtime_error("IntoMLIR verification failure");
  }

  IVLOG(1, "Doing some passes");
  mlir::PassManager pm(&context, true);
  pm.addPass(mlir::createCSEPass());
  codegen::proto::MLIR_PadPass options;
  pm.addPass(CreatePass<PaddingPass>(options));
  if (failed(pm.run(*module))) {
    module->dump();
    throw std::runtime_error("MLIR passes failure");
  }

  IVLOG(1, "Writing out module");
  auto moduleOp = *module;
  auto module_str = mlir::debugString(moduleOp);
  IVLOG(2, module_str);

  IVLOG(1, "Parsing it back in");
  auto new_module = parseSourceString(module_str, &context);
  if (!new_module) {
    throw std::runtime_error("Unable to parse");
  }

  IVLOG(1, "Converting the other way");
  auto stripe2 = FromMLIR(*new_module);

  IVLOG(2, "New version:");
  IVLOG(2, *stripe2->entry);

  // require textually perfect round trip
  EXPECT_THAT(to_string(*stripe2->entry), LinesEq(to_string(*stripe->entry)));
}

TEST_P(TranslateTest, Conv2dBnRelu) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {16, 112, 112, 64}, "I");
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 64, 128}, "K");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {128}, "B");
  auto S = Placeholder(PLAIDML_DATA_FLOAT32, {128}, "S");
  auto O = plaidml::op::convolution(  //
      I,                              // I_or_O
      K,                              // F_or_O
      {2, 2},                         // strides
      {1, 1},                         // dilations
      {1, 1},                         // data_dilations
      {},                             // filter_shape
      1,                              // groups
      "explicit",                     // autopad_mode
      {3, 3},                         // manual_padding
      "nxc",                          // input_layout
      "xck",                          // filter_layout
      "none",                         // group_layout
      false,                          // winograd_allowed
      "",                             // name
      "ungrouped",                    // autogroup_mode
      "none",                         // deriv_mode
      {});                            // result_shape
  auto R = plaidml::op::relu((O + B) * S);
  Program program("Conv2dBnRelu", {R});
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, MaxPool2d) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 64, 64, 3}, "I");
  auto O = plaidml::op::pool(I, "max", {2, 2}, {1, 1}, "none", {1, 2}, "nwc", true, true);
  Program program("pool", {O});
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, Softmax) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {64, 64}, "A");
  Program program("softmax", {plaidml::op::softmax(A, 1)});
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, EltwiseAdd) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {16, 16}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {16, 16}, "B");
  Program program("eltwise_add", {A + B});
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, ArgMax) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  Program program("argmax", {plaidml::op::argmax(I)});
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, Cast) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  Program program("cast", {as_bool(I)});
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, Scalar) {
  LogicalShape shape(PLAIDML_DATA_FLOAT32, {16, 64, 64, 32});
  auto S = Placeholder(PLAIDML_DATA_FLOAT32, {}, "S");
  Program program("scalar", {S});
  RunTest(program, GetParam());
}

TEST_P(TranslateTest, DoubleDot) {
  using plaidml::op::dot;
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {20, 30}, "B");
  auto C = Placeholder(PLAIDML_DATA_FLOAT32, {30, 40}, "C");
  Program program("double_dot", {dot(dot(A, B), C)});
  RunTest(program, GetParam());
}

INSTANTIATE_TEST_CASE_P(NonTrivialLocs, TranslateTest, ::testing::Values(false, true));
