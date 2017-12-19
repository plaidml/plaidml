#include "tile/lang/gen_contract.h"

#include <gtest/gtest.h>

#include "tile/lang/compile.h"
#include "tile/lang/emitc.h"
#include "tile/lang/parser.h"
#include "tile/lang/sembuilder.h"

namespace vertexai {
namespace tile {
namespace lang {

struct TestParam {
  HardwareSettings settings;
  std::string program;
  ShapeMap inputs;
  ShapeMap outputs;
  std::shared_ptr<sem::Function> expected;
};

void RunTest(const TestParam& param) {
  Parser parser;
  auto program = parser.Parse(param.program);
  auto result = GenerateProgram(program, param.inputs, param.outputs, param.settings, "test");

  lang::EmitDebug actual;
  actual.Visit(*result.kernels[0].kfunc);

  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Generic debug kernel:";
    VLOG(4) << actual.str();
  }

  lang::EmitDebug expected;
  expected.Visit(*param.expected);

  EXPECT_EQ(expected.str(), actual.str());
}

HardwareSettings TestGPU() {
  HardwareSettings settings;
  settings.threads = 256;
  settings.vec_size = 1;
  settings.use_global = false;
  settings.mem_width = 32;
  settings.max_mem = 18 * 1024;
  settings.max_regs = 18 * 1024;
  settings.goal_groups = 20;
  settings.goal_flops_per_byte = 20;
  settings.goal_dimension_sizes = {1024, 1024, 1024};
  return settings;
}

TEST(GenContractTest, PiecewiseMultiply) {
  using namespace sem::builder;  // NOLINT
  sem::Type index_type{sem::Type::INDEX};
  sem::Type in_float_type{sem::Type::POINTER_CONST, DataType::FLOAT32};
  sem::Type out_float_type{sem::Type::POINTER_MUT, DataType::FLOAT32};
  sem::Type float_type{sem::Type::VALUE, lang::DataType::FLOAT32};
  RunTest(TestParam{
      TestGPU(),  // settings
      R"(
        function (A[X, Y], B[Y, X]) -> (C) {
          C[x, y : X, Y] = +(A[x, y] * B[y, x]);
        }
      )",         // program
      {
          // inputs
          {"A", SimpleShape(DataType::FLOAT32, {4, 4})},
          {"B", SimpleShape(DataType::FLOAT32, {4, 4})},
      },
      {
          // outputs
          {"C", SimpleShape(DataType::FLOAT32, {4, 4})},
      },
      _Function(            // expected
          "kernel_test_0",  // Function.name
          sem::Type{},      // Function.ret
          {
              // Function.params
              {out_float_type, "C"},   // C
              {in_float_type, "in1"},  // A
              {in_float_type, "in2"},  // B
          },
          {
              _Declare(index_type, "tid", _Index(sem::IndexExpr::LOCAL, 0)),
              _Declare({sem::Type::VALUE, lang::DataType::FLOAT32, 1, 1}, "agg",
                       _LimitConst(sem::LimitConst::ZERO, lang::DataType::FLOAT32)),
              _Declare({sem::Type::VALUE, lang::DataType::FLOAT32, 1, 16}, "in1_shared", nullptr),
              _Declare({sem::Type::VALUE, lang::DataType::FLOAT32, 1, 16}, "in2_shared", nullptr),  // EOL
              _Block({
                  _Block({
                      _Declare(index_type, "y_x_tid", (_("tid") % 16)),
                      _("in1_shared")[_("y_x_tid")] = _("in1")[_Clamp(_("y_x_tid"), _Const(0), _Const(15))],
                  }),
                  _Block({
                      _Declare(index_type, "x_y_tid", (_("tid") % 16)),
                      _("in2_shared")[_("x_y_tid")] = _("in2")[_Clamp(_("x_y_tid"), _Const(0), _Const(15))],
                  }),
                  _Barrier(),                                           // EOL
                  _Declare(index_type, "y_tid", (_("tid") % 4)),        // EOL
                  _Declare(index_type, "x_tid", ((_("tid") / 4) % 4)),  // EOL
                  _Declare(float_type, "val1", _Cast(float_type, _("in1_shared")[_("y_tid") + (4 * _("x_tid"))])),
                  _Declare(float_type, "val2", _Cast(float_type, _("in2_shared")[_("x_tid") + (4 * _("y_tid"))])),
                  _Declare(float_type, "pre_agg", _Cast(float_type, (_("val2") * _("val1")))),  // EOL
                  _("agg")[_Const(0)] = (_("agg")[_Const(0)] + _("pre_agg")),                   // EOL
                  _Barrier(),                                                                   // EOL
              }),
              _Declare(index_type, "y_tid", (_("tid") % 4)),                      // EOL
              _Declare(index_type, "x_tid", ((_("tid") / 4) % 4)),                // EOL
              _Declare(float_type, "LC", _("agg")[_Const(0)]),                    // EOL
              _Declare(index_type, "gout_idx", ((4 * _("x_tid")) + _("y_tid"))),  // EOL
              _("C")[_("gout_idx")] = _("LC"),                                    // EOL
          }),
  });
}

TEST(GenContractTest, SmallMatMul) {
  using namespace sem::builder;  // NOLINT
  sem::Type index_type{sem::Type::INDEX};
  sem::Type in_float_type{sem::Type::POINTER_CONST, DataType::FLOAT32};
  sem::Type out_float_type{sem::Type::POINTER_MUT, DataType::FLOAT32};
  sem::Type float_type{sem::Type::VALUE, lang::DataType::FLOAT32};
  RunTest(TestParam{
      TestGPU(),  // settings
      R"(
        function (A[I, K], B[K, J]) -> (C) {
          C[i, j : I, J] = +(A[i, k] * B[k, j]);
        }
      )",         // program
      {
          // inputs
          {"A", TensorShape(DataType::FLOAT32, {{4, 4}, {4, 4}})},
          {"B", TensorShape(DataType::FLOAT32, {{4, 4}, {4, 4}})},
      },
      {
          // outputs
          {"C", TensorShape(DataType::FLOAT32, {{4, 4}, {4, 4}})},
      },
      _Function(            // expected
          "kernel_test_0",  // Function.name
          sem::Type{},      // Function.ret
          {
              // Function.params
              {out_float_type, "C"},   // C
              {in_float_type, "in1"},  // A
              {in_float_type, "in2"},  // B
          },
          {
              _Declare(index_type, "tid", _Index(sem::IndexExpr::LOCAL, 0)),
              _Declare({sem::Type::VALUE, lang::DataType::FLOAT32, 1, 1}, "agg",
                       _LimitConst(sem::LimitConst::ZERO, lang::DataType::FLOAT32)),
              _Declare({sem::Type::VALUE, lang::DataType::FLOAT32, 1, 7}, "in1_shared", nullptr),
              _Declare({sem::Type::VALUE, lang::DataType::FLOAT32, 1, 7}, "in2_shared", nullptr),  // EOL
              _For("k_gid", 1, 4,
                   _Block({
                       _Block({
                           _Declare(index_type, "gbase", (_("k_gid") * 4)),
                           _Declare(index_type, "i_k_tid", (_("tid") % 8)),
                           _Declare(index_type, "i_k_cond", (_("i_k_tid") < 7)),
                           _If(_("i_k_cond"),
                               _Block({
                                   _If((_("tid") < 8),
                                       _Block({
                                           _Declare(index_type, "gidx", (_("gbase") + (_Const(4) * _("i_k_tid")))),
                                           _("in1_shared")[_("i_k_tid")] =
                                               _("in1")[_Clamp(_("gidx"), _Const(0), _Const(24))],
                                       })),
                               })),
                       }),
                       _Block({
                           _Declare(index_type, "gbase", (_("k_gid") * 4)),
                           _Declare(index_type, "j_k_tid", (_("tid") % 8)),
                           _Declare(index_type, "j_k_cond", (_("j_k_tid") < 7)),
                           _If(_("j_k_cond"),
                               _Block({
                                   _If((_("tid") < 8),
                                       _Block({
                                           _Declare(index_type, "gidx", (_("gbase") + (_Const(4) * _("j_k_tid")))),
                                           _("in2_shared")[_("j_k_tid")] =
                                               _("in2")[_Clamp(_("gidx"), _Const(0), _Const(24))],
                                       })),
                               })),
                       }),
                       _Barrier(),                                            // EOL
                       _Declare(index_type, "k_tid", ((_("tid") / 16) % 4)),  // EOL
                       _Declare(index_type, "j_tid", (_("tid") % 4)),         // EOL
                       _Declare(index_type, "i_tid", ((_("tid") / 4) % 4)),   // EOL
                       _Declare(float_type, "val1", _Cast(float_type, _("in1_shared")[_("i_tid") + _("k_tid")])),
                       _Declare(float_type, "val2", _Cast(float_type, _("in2_shared")[_("j_tid") + _("k_tid")])),
                       _Declare(float_type, "pre_agg", _Cast(float_type, (_("val2") * _("val1")))),  // EOL
                       _("agg")[_Const(0)] = (_("agg")[_Const(0)] + _("pre_agg")),                   // EOL
                       _Barrier(),                                                                   // EOL
                   })),
              _Declare({sem::Type::VALUE, lang::DataType::FLOAT32, 1, 64}, "merge_shared", nullptr),  // EOL
              _Block({
                  _("merge_shared")[_("tid")] = _("agg")[_Const(0)],
                  _Barrier(),  // EOL
                  _If(_("tid") < 32,
                      _Block({
                          _("merge_shared")[_("tid")] = _("merge_shared")[_("tid")] + _("merge_shared")[_("tid") + 32],
                      })),
                  _Barrier(),  // EOL
                  _If(_("tid") < 16,
                      _Block({
                          _("merge_shared")[_("tid")] = _("merge_shared")[_("tid")] + _("merge_shared")[_("tid") + 16],
                      })),
                  _Barrier(),         // EOL
                  _If(_("tid") < 16,  // EOL
                      _Block({
                          _("agg")[_Const(0)] = _("merge_shared")[_("tid")]  // EOL
                      })),
              }),
              _Declare(index_type, "j_tid", (_("tid") % 4)),        // EOL
              _Declare(index_type, "i_tid", ((_("tid") / 4) % 4)),  // EOL
              _If(_("tid") < 16,                                    // EOL
                  _Block({
                      _Declare(float_type, "LC", _("agg")[_Const(0)]),                    // EOL
                      _Declare(index_type, "gout_idx", ((4 * _("i_tid")) + _("j_tid"))),  // EOL
                      _("C")[_("gout_idx")] = _("LC"),                                    // EOL
                  })),
          }),
  });
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
