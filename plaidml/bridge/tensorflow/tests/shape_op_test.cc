// Tests that show HLO Module conversion to PlaidML Program.

// #include "plaidml/bridge/tensorflow/tests/shape_op_test.h.inc"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "plaidml/bridge/tensorflow/tests/codegen_test.h"

namespace xla {
namespace plaidml {
namespace {

struct ShapeTestSpec {
  PrimitiveType primitive_type;
};

string ShapeTestSpecToString(const ::testing::TestParamInfo<ShapeTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLShapeOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<ShapeTestSpec> {};

// TODO: rewrite this test as an actual python-based unit tests, so we can use the tf python bindings directly.
// TEST_P(PlaidMLShapeOperationTest, BroadcastTest) {
//   VLOG(0) << "Testing generated examples";

//   for (std::size_t i = 0; i < broadcast_modules.size(); ++i) {
//     std::string set_des = broadcast_descriptions[i];
//     VLOG(0) << "Testing set " << i << ": " << set_des;
//     TestCaseVal inputs = broadcast_is[i];
//     TestCaseVal results = broadcast_os[i];
//     std::string module_text = broadcast_modules[i];

//     TestCasePairs testcase_pairs = {{inputs, results}};

//     ShapeTestSpec spec = GetParam();

//     HloModuleConfig cfg;

//     std::unique_ptr<VerifiedHloModule> hlo_module =
//         absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

//     hlo_module->ParseHloStringAndVerifyModule(module_text);

//     CompileAndCheck(std::move(hlo_module),  testcase_pairs);
//   }
// }

// TODO: rewrite this test as an actual python-based unit tests, so we can use the tf python bindings directly.
// TEST_P(PlaidMLShapeOperationTest, ReshapeTest) {
//   VLOG(0) << "Testing generated examples";

//   for (std::size_t i = 0; i < reshape_modules.size(); ++i) {
//     std::string set_des = reshape_descriptions[i];
//     VLOG(0) << "Testing set " << i << ": " << set_des;
//     TestCaseVal inputs = reshape_is[i];
//     TestCaseVal results = reshape_os[i];
//     std::string module_text = reshape_modules[i];

//     TestCasePairs testcase_pairs = {{inputs, results}};

//     ShapeTestSpec spec = GetParam();

//     HloModuleConfig cfg;

//     std::unique_ptr<VerifiedHloModule> hlo_module =
//         absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

//     hlo_module->ParseHloStringAndVerifyModule(module_text);

//     CompileAndCheck(std::move(hlo_module),  testcase_pairs);
//   }
// }

// TODO: rewrite this test as an actual python-based unit tests, so we can use the tf python bindings directly.
// TEST_P(PlaidMLShapeOperationTest, PadTest) {
//   VLOG(0) << "Testing generated examples";

//   for (std::size_t i = 0; i < pad_modules.size(); ++i) {
//     std::string set_des = pad_descriptions[i];
//     VLOG(0) << "Testing set " << i << ": " << set_des;
//     TestCaseVal inputs = pad_is[i];
//     TestCaseVal results = pad_os[i];
//     std::string module_text = pad_modules[i];

//     TestCasePairs testcase_pairs = {{inputs, results}};

//     ShapeTestSpec spec = GetParam();

//     HloModuleConfig cfg;

//     std::unique_ptr<VerifiedHloModule> hlo_module =
//         absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

//     hlo_module->ParseHloStringAndVerifyModule(module_text);

//     CompileAndCheck(std::move(hlo_module),  testcase_pairs);
//   }
// }

// TODO: rewrite this test as an actual python-based unit tests, so we can use the tf python bindings directly.
// TEST_P(PlaidMLShapeOperationTest, SliceTest) {
//   VLOG(0) << "Testing generated examples";

//   for (std::size_t i = 0; i < slice_modules.size(); ++i) {
//     std::string set_des = slice_descriptions[i];
//     VLOG(0) << "Testing set " << i << ": " << set_des;
//     TestCaseVal inputs = slice_is[i];
//     TestCaseVal results = slice_os[i];
//     std::string module_text = slice_modules[i];

//     TestCasePairs testcase_pairs = {{inputs, results}};

//     ShapeTestSpec spec = GetParam();

//     HloModuleConfig cfg;

//     std::unique_ptr<VerifiedHloModule> hlo_module =
//         absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

//     hlo_module->ParseHloStringAndVerifyModule(module_text);

//     CompileAndCheck(std::move(hlo_module),  testcase_pairs);
//   }
// }

// TODO: rewrite this test as an actual python-based unit tests, so we can use the tf python bindings directly.
// TEST_P(PlaidMLShapeOperationTest, TransposeTest) {
//   VLOG(0) << "Testing generated examples";

//   for (std::size_t i = 0; i < transpose_modules.size(); ++i) {
//     std::string set_des = transpose_descriptions[i];
//     VLOG(0) << "Testing set " << i << ": " << set_des;
//     TestCaseVal inputs = transpose_is[i];
//     TestCaseVal results = transpose_os[i];
//     std::string module_text = transpose_modules[i];

//     TestCasePairs testcase_pairs = {{inputs, results}};

//     ShapeTestSpec spec = GetParam();

//     HloModuleConfig cfg;

//     std::unique_ptr<VerifiedHloModule> hlo_module =
//         absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

//     hlo_module->ParseHloStringAndVerifyModule(module_text);

//     CompileAndCheck(std::move(hlo_module),  testcase_pairs);
//   }
// }

std::vector<ShapeTestSpec> GetShapeTestCases() {
  std::vector<ShapeTestSpec> result;
  result.push_back({F32});
  result.push_back({F64});
  return result;
}

INSTANTIATE_TEST_SUITE_P(All, PlaidMLShapeOperationTest, ::testing::ValuesIn(GetShapeTestCases()),
                         ShapeTestSpecToString);

}  // namespace
}  // namespace plaidml
}  // namespace xla
