// Tests that show HLO Module conversion to PlaidML Program.

#include "plaidml/bridge/tensorflow/tests/shape_op_test.h.inc"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "absl/strings/str_cat.h"
#include "plaidml/bridge/tensorflow/service/compiler.h"
#include "plaidml/bridge/tensorflow/tests/codegen_test.h"
#include "plaidml/bridge/tensorflow/tests/filecheck.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

using ::plaidml::edsl::TensorBuffers;

namespace xla {
namespace plaidml {
namespace {

using TestCaseVal = std::vector<std::vector<float>>;
using TestCasePairs = std::map<TestCaseVal, TestCaseVal>;

struct ShapeTestSpec {
  PrimitiveType primitive_type;
  string filecheck_lines;
};

string ShapeTestSpecToString(const ::testing::TestParamInfo<ShapeTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class PlaidMLShapeOperationTest : public PlaidMLCodegenTest, public ::testing::WithParamInterface<ShapeTestSpec> {
 protected:
  Status CompileAndCheck(std::unique_ptr<HloModule> hlo_module, const string& filecheck_lines,
                         const TestCasePairs& testcase_pairs) {
    auto program = CompileToProgram(std::move(hlo_module));

    StatusOr<bool> fc_result = RunFileCheck(program->str(), filecheck_lines);

    // TF_ASSERT_OK(fc_result.status());
    EXPECT_TRUE(fc_result.ValueOrDie());

    VLOG(2) << "Evaluating results";

    for (auto pair : testcase_pairs) {
      TensorBuffers inp;
      TensorBuffers exp;

      auto program_inputs = program->inputs();
      auto tcp_inputs = pair.first;

      if (tcp_inputs.size() != program_inputs.size()) {
        VLOG(1) << "Found mismatch in input sizes: tcp " << tcp_inputs.size() << " program " << program_inputs.size();
      }

      for (auto i = 0; i < program_inputs.size(); i++) {
        VLOG(1) << "Adding TestCaseInput " << i;
        inp.insert(std::make_pair(program_inputs[i].tensor, pair.first[i]));
      }

      auto program_outputs = program->outputs();
      auto tcp_outputs = pair.second;

      if (tcp_outputs.size() != program_outputs.size()) {
        VLOG(1) << "Found mismatch in output sizes: tcp " << tcp_outputs.size() << " program "
                << program_outputs.size();
      }

      for (auto i = 0; i < program_outputs.size(); i++) {
        VLOG(1) << "Adding TestCaseOutput " << i;
        exp.insert(std::make_pair(program_outputs[i].tensor, pair.second[i]));
      }

      VLOG(2) << "Calling checkProgram";

      checkProgram(*program, inp, exp);
    }
    return Status::OK();
  }
};

TEST_P(PlaidMLShapeOperationTest, BroadcastTest) {
  VLOG(0) << "Testing generated examples";

  for (std::size_t i = 0; i < broadcast_modules.size(); ++i) {
    std::string set_des = broadcast_descriptions[i];
    VLOG(0) << "Testing set " << i << ": " << set_des;
    TestCaseVal inputs = broadcast_is[i];
    TestCaseVal results = broadcast_os[i];
    std::string module_text = broadcast_modules[i];

    TestCasePairs testcase_pairs = {{inputs, results}};

    ShapeTestSpec spec = GetParam();
    auto fcheck_lines = spec.filecheck_lines;
    std::string match = "> {";
    fcheck_lines.insert(fcheck_lines.find(match) + 4, "CHECK: %{{.*}} = tile.contract assign\n");

    HloModuleConfig cfg;

    std::unique_ptr<VerifiedHloModule> hlo_module =
        absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

    hlo_module->ParseHloStringAndVerifyModule(module_text);

    CompileAndCheck(std::move(hlo_module), fcheck_lines, testcase_pairs);
  }
}

TEST_P(PlaidMLShapeOperationTest, ReshapeTest) {
  VLOG(0) << "Testing generated examples";

  for (std::size_t i = 0; i < reshape_modules.size(); ++i) {
    std::string set_des = reshape_descriptions[i];
    VLOG(0) << "Testing set " << i << ": " << set_des;
    TestCaseVal inputs = reshape_is[i];
    TestCaseVal results = reshape_os[i];
    std::string module_text = reshape_modules[i];

    TestCasePairs testcase_pairs = {{inputs, results}};

    ShapeTestSpec spec = GetParam();
    auto fcheck_lines = spec.filecheck_lines;
    std::string match = "> {";
    fcheck_lines.insert(fcheck_lines.find(match) + 4, "CHECK: %{{.*}} = \"tile.reshape\"\n");

    HloModuleConfig cfg;

    std::unique_ptr<VerifiedHloModule> hlo_module =
        absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

    hlo_module->ParseHloStringAndVerifyModule(module_text);

    CompileAndCheck(std::move(hlo_module), fcheck_lines, testcase_pairs);
  }
}

TEST_P(PlaidMLShapeOperationTest, PadTest) {
  VLOG(0) << "Testing generated examples";

  for (std::size_t i = 0; i < pad_modules.size(); ++i) {
    std::string set_des = pad_descriptions[i];
    VLOG(0) << "Testing set " << i << ": " << set_des;
    TestCaseVal inputs = pad_is[i];
    TestCaseVal results = pad_os[i];
    std::string module_text = pad_modules[i];

    TestCasePairs testcase_pairs = {{inputs, results}};

    ShapeTestSpec spec = GetParam();
    auto fcheck_lines = spec.filecheck_lines;
    std::string match = "> {";
    fcheck_lines.insert(fcheck_lines.find(match) + 4, "CHECK: %{{.*}} = tile.contract assign\n");

    HloModuleConfig cfg;

    std::unique_ptr<VerifiedHloModule> hlo_module =
        absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

    hlo_module->ParseHloStringAndVerifyModule(module_text);

    CompileAndCheck(std::move(hlo_module), fcheck_lines, testcase_pairs);
  }
}

TEST_P(PlaidMLShapeOperationTest, SliceTest) {
  VLOG(0) << "Testing generated examples";

  for (std::size_t i = 0; i < slice_modules.size(); ++i) {
    std::string set_des = slice_descriptions[i];
    VLOG(0) << "Testing set " << i << ": " << set_des;
    TestCaseVal inputs = slice_is[i];
    TestCaseVal results = slice_os[i];
    std::string module_text = slice_modules[i];

    TestCasePairs testcase_pairs = {{inputs, results}};

    ShapeTestSpec spec = GetParam();
    auto fcheck_lines = spec.filecheck_lines;
    std::string match = "> {";
    fcheck_lines.insert(fcheck_lines.find(match) + 4, "CHECK: %{{.*}} = tile.contract assign\n");

    HloModuleConfig cfg;

    std::unique_ptr<VerifiedHloModule> hlo_module =
        absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

    hlo_module->ParseHloStringAndVerifyModule(module_text);

    CompileAndCheck(std::move(hlo_module), fcheck_lines, testcase_pairs);
  }
}

TEST_P(PlaidMLShapeOperationTest, TransposeTest) {
  VLOG(0) << "Testing generated examples";

  for (std::size_t i = 0; i < transpose_modules.size(); ++i) {
    std::string set_des = transpose_descriptions[i];
    VLOG(0) << "Testing set " << i << ": " << set_des;
    TestCaseVal inputs = transpose_is[i];
    TestCaseVal results = transpose_os[i];
    std::string module_text = transpose_modules[i];

    TestCasePairs testcase_pairs = {{inputs, results}};

    ShapeTestSpec spec = GetParam();
    auto fcheck_lines = spec.filecheck_lines;
    std::string match = "> {";
    fcheck_lines.insert(fcheck_lines.find(match) + 4, "CHECK: %{{.*}} = tile.contract assign\n");

    HloModuleConfig cfg;

    std::unique_ptr<VerifiedHloModule> hlo_module =
        absl::make_unique<VerifiedHloModule>("module", cfg, false, false, nullptr);

    hlo_module->ParseHloStringAndVerifyModule(module_text);

    CompileAndCheck(std::move(hlo_module), fcheck_lines, testcase_pairs);
  }
}

std::vector<ShapeTestSpec> GetShapeTestCases() {
  std::vector<ShapeTestSpec> result;
  auto check_str = R"#(
                        CHECK: func @hlo_module{{.*}}%[[I:.*]]: tensor<[[is:.*]]x[[prec:.*]]>) -> tensor<[[os:.*]]x[[prec]]> {
                        CHECK: return %{{.*}} : tensor<[[os.*]]x[[prec]]>
                    )#";
  result.push_back({F32, check_str});
  result.push_back({F64, check_str});
  return result;
}

/**/
// TODO: INSTANTIATE_TEST_CASE_P was deprecated in favor for INSTANTIATE_TEST_SUITE_P, but the version of gtest that
// bazel links in is looking for INSTANTIATE_TEST_CASE_P right now.
INSTANTIATE_TEST_CASE_P(All, PlaidMLShapeOperationTest, ::testing::ValuesIn(GetShapeTestCases()),
                        ShapeTestSpecToString);
/**/
}  // namespace
}  // namespace plaidml
}  // namespace xla
