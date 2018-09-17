#include "tile/lang/simplifier.h"

#include <gtest/gtest.h>

#include "tile/lang/sembuilder.h"
#include "tile/lang/semprinter.h"

namespace vertexai {
namespace tile {
namespace lang {

struct TestParam {
  std::shared_ptr<sem::Function> before;
  std::shared_ptr<sem::Function> after;
};

class SimplifierTest : public ::testing::TestWithParam<TestParam> {};

TEST_P(SimplifierTest, Compare) {
  auto param = GetParam();

  KernelInfo ki;
  ki.kfunc = param.before;

  std::vector<KernelInfo> kernels{ki};
  Simplify(kernels);

  sem::Print actual(*ki.kfunc);

  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Generic debug kernel after simplification:";
    VLOG(4) << actual.str();
  }

  sem::Print expected(*param.after);

  EXPECT_EQ(actual.str(), expected.str());
}

TestParam Basic() {
  using namespace sem::builder;  // NOLINT

  auto A = _("A");
  auto B = _("B");
  auto C = _("C");
  auto tid = _("tid");
  auto i1_i2 = _("i1_i2");
  auto i1_i2_gid = _("i1_i2_gid");
  auto i1_i2_tid = _("i1_i2_tid");
  auto i1_i2_lid = _("i1_i2_lid");
  auto gout_idx = _("gout_idx");
  auto LA = _("LA");
  auto LB = _("LB");
  auto LC = _("LC");

  sem::Type short_type{sem::Type::VALUE, lang::DataType::INT16};
  sem::Type index_type{sem::Type::INDEX};

  // void kernel__0(short* C, const short* A, const short* B) {
  //   int tid = get_local_id(0);
  //   int i1_i2_gid = 0;
  //   int i1_i2_tid = ((tid / 1) % 16);
  //   int i1_i2_lid = 0;
  //   int i1_i2 = ((16 * i1_i2_lid) + i1_i2_tid);
  //   int gout_idx = (0 + (1 * (i1_i2_gid + i1_i2)));
  //   short LA = A[gout_idx];
  //   short LB = B[gout_idx];
  //   short LC = (((short)LA) + ((short)LB));
  //   C[gout_idx] = LC;
  // }
  auto before = _Block({});
  before->append(_Declare(index_type, "tid", _Index(sem::IndexExpr::LOCAL, 0)));
  before->append(_DeclareConst(index_type, "i1_i2_gid", 0));
  before->append(_Declare(index_type, "i1_i2_tid", (tid / 1) % 16));
  before->append(_DeclareConst(index_type, "i1_i2_lid", 0));
  before->append(_Declare(index_type, "i1_i2", (16 * i1_i2_lid) + i1_i2_tid));
  before->append(_Declare(index_type, "gout_idx", (0 + (1 * (i1_i2_gid + i1_i2)))));
  before->append(_Declare(short_type, "LA", A[gout_idx]));
  before->append(_Declare(short_type, "LB", B[gout_idx]));
  before->append(_Declare(short_type, "LC", _Cast(short_type, LA) + _Cast(short_type, LB)));
  before->append(C[gout_idx] = LC);

  // __kernel void kernel__0(__global short* C, __global const short* A, __global const short* B) {
  //   int tid = get_local_id(0);
  //   int i1_i2_tid = (tid % 16);
  //   short LA = A[i1_i2_tid];
  //   short LB = B[i1_i2_tid];
  //   short LC = (LA + LB);
  //   C[i1_i2_tid] = LC;
  // }
  auto after = _Block({});
  after->append(_Declare(index_type, "tid", _Index(sem::IndexExpr::LOCAL, 0)));
  after->append(_Declare(index_type, "i1_i2_tid", (tid % 16)));
  after->append(_Declare(short_type, "LA", A[i1_i2_tid]));
  after->append(_Declare(short_type, "LB", B[i1_i2_tid]));
  after->append(_Declare(short_type, "LC", _Cast(short_type, LA) + _Cast(short_type, LB)));
  after->append(C[i1_i2_tid] = LC);

  auto fn_before = _Function("kernel", sem::Type{}, {}, {before});
  auto fn_after = _Function("kernel", sem::Type{}, {}, {after});
  return TestParam{fn_before, fn_after};
}

TestParam Contraction() {
  using namespace sem::builder;  // NOLINT

  sem::Type short_type{sem::Type::VALUE, lang::DataType::INT16};
  sem::Type index_type{sem::Type::INDEX};

  // void kernel__0(short* C, const short* in1, const short* in2)
  // {
  //   int tid = get_local_id(0);
  //   short agg[1] = {0, };
  //   short in1_shared[16];
  //   short in2_shared[16];
  //   int y_gid = 0;
  //   int x_gid = 0;
  //   {
  //     {
  //       int gbase = ((0 + (y_gid * 1)) + (x_gid * 4));
  //       int y_x_tid = ((tid / 1) % 16);
  //       int y_x_lid = 0;
  //       int y_x = ((16 * y_x_lid) + y_x_tid);
  //       int lidx = (0 + (1 * y_x));
  //       int gidx = (gbase + (1 * y_x));
  //       in1_shared[lidx] = in1[clamp(gidx, 0, 15)];
  //     }
  //     {
  //       int gbase = ((0 + (x_gid * 1)) + (y_gid * 4));
  //       int x_y_tid = ((tid / 1) % 16);
  //       int x_y_lid = 0;
  //       int x_y = ((16 * x_y_lid) + x_y_tid);
  //       int lidx = (0 + (1 * x_y));
  //       int gidx = (gbase + (1 * x_y));
  //       in2_shared[lidx] = in2[clamp(gidx, 0, 15)];
  //     }
  //     barrier();
  //     int y_tid = ((tid / 1) % 4);
  //     int x_tid = ((tid / 4) % 4);
  //     int y_lid = 0;
  //     int y = ((4 * y_lid) + y_tid);
  //     int x_lid = 0;
  //     int x = ((4 * x_lid) + x_tid);
  //     short val1 = ((short)in1_shared[((0 + (1 * y)) + (4 * x))]);
  //     short val2 = ((short)in2_shared[((0 + (1 * x)) + (4 * y))]);
  //     short pre_agg = ((short)(val2 * val1));
  //     agg[((0 + (y_lid * 1)) + (x_lid * 1))] = (agg[((0 + (y_lid * 1)) + (x_lid * 1))] + pre_agg);
  //     barrier();
  //   }
  //   int y_tid = ((tid / 1) % 4);
  //   int x_tid = ((tid / 4) % 4);
  //   int y_lid = 0;
  //   int y = ((4 * y_lid) + y_tid);
  //   int x_lid = 0;
  //   int x = ((4 * x_lid) + x_tid);
  //   short LC = agg[((0 + (y_lid * 1)) + (x_lid * 1))];
  //   int gout_idx = ((0 + (4 * (x_gid + x))) + (1 * (y_gid + y)));
  //   C[gout_idx] = LC;
  // }
  auto before = _Block({});
  {
    auto C = _("C");
    auto in1 = _("in1");
    auto in2 = _("in2");
    before->append(_Declare(index_type, "tid", _Index(sem::IndexExpr::LOCAL, 0)));
    auto tid = _("tid");
    before->append(_Declare({sem::Type::VALUE, lang::DataType::INT16, 1, 1}, "agg",
                            _LimitConst(sem::LimitConst::ZERO, lang::DataType::INT16)));
    auto agg = _("agg");
    before->append(_Declare({sem::Type::VALUE, lang::DataType::INT16, 1, 16}, "in1_shared", nullptr));
    auto in1_shared = _("in1_shared");
    before->append(_Declare({sem::Type::VALUE, lang::DataType::INT16, 1, 16}, "in2_shared", nullptr));
    auto in2_shared = _("in2_shared");
    before->append(_DeclareConst(index_type, "y_gid", 0));
    auto y_gid = _("y_gid");
    before->append(_DeclareConst(index_type, "x_gid", 0));
    auto x_gid = _("x_gid");
    {
      auto before_b1 = _Block({});
      {
        auto before_b1_b1 = _Block({});
        before_b1_b1->append(_Declare(index_type, "gbase", (0 + (y_gid * 1)) + (x_gid * 4)));
        auto gbase = _("gbase");
        before_b1_b1->append(_Declare(index_type, "y_x_tid", ((tid / 1) % 16)));
        auto y_x_tid = _("y_x_tid");
        before_b1_b1->append(_DeclareConst(index_type, "y_x_lid", 0));
        auto y_x_lid = _("y_x_lid");
        before_b1_b1->append(_Declare(index_type, "y_x", ((16 * y_x_lid) + y_x_tid)));
        auto y_x = _("y_x");
        before_b1_b1->append(_Declare(index_type, "lidx", (0 + (1 * y_x))));
        auto lidx = _("lidx");
        before_b1_b1->append(_Declare(index_type, "gidx", (gbase + (1 * y_x))));
        auto gidx = _("gidx");
        before_b1_b1->append(in1_shared[lidx] = in1[_Clamp(gidx, _Const(0), _Const(15))]);
        before_b1->push_back(before_b1_b1);
      }
      {
        auto before_b1_b2 = _Block({});
        before_b1_b2->append(_Declare(index_type, "gbase", (0 + (x_gid * 1)) + (y_gid * 4)));
        auto gbase = _("gbase");
        before_b1_b2->append(_Declare(index_type, "x_y_tid", ((tid / 1) % 16)));
        auto x_y_tid = _("x_y_tid");
        before_b1_b2->append(_DeclareConst(index_type, "x_y_lid", 0));
        auto x_y_lid = _("x_y_lid");
        before_b1_b2->append(_Declare(index_type, "x_y", ((16 * x_y_lid) + x_y_tid)));
        auto x_y = _("x_y");
        before_b1_b2->append(_Declare(index_type, "lidx", (0 + (1 * x_y))));
        auto lidx = _("lidx");
        before_b1_b2->append(_Declare(index_type, "gidx", (gbase + (1 * x_y))));
        auto gidx = _("gidx");
        before_b1_b2->append(in2_shared[_("lidx")] = in2[_Clamp(_("gidx"), _Const(0), _Const(15))]);
        before_b1->push_back(before_b1_b2);
      }
      before_b1->append(_Barrier());
      before_b1->append(_Declare(index_type, "y_tid", ((tid / 1) % 4)));
      auto y_tid = _("y_tid");
      before_b1->append(_Declare(index_type, "x_tid", ((tid / 4) % 4)));
      auto x_tid = _("x_tid");
      before_b1->append(_DeclareConst(index_type, "y_lid", 0));
      auto y_lid = _("y_lid");
      before_b1->append(_Declare(index_type, "y", ((4 * y_lid) + y_tid)));
      auto y = _("y");
      before_b1->append(_DeclareConst(index_type, "x_lid", 0));
      auto x_lid = _("x_lid");
      before_b1->append(_Declare(index_type, "x", ((4 * x_lid) + x_tid)));
      auto x = _("x");
      before_b1->append(_Declare(short_type, "val1", _Cast(short_type, in1_shared[((0 + (1 * y)) + (4 * x))])));
      auto val1 = _("val1");
      before_b1->append(_Declare(short_type, "val2", _Cast(short_type, in2_shared[((0 + (1 * x)) + (4 * y))])));
      auto val2 = _("val2");
      before_b1->append(_Declare(short_type, "pre_agg", _Cast(short_type, (val2 * val1))));
      auto pre_agg = _("pre_agg");
      before_b1->append(agg[((0 + (y_lid * 1)) + (x_lid * 1))] = (agg[((0 + (y_lid * 1)) + (x_lid * 1))] + pre_agg));
      before_b1->append(_Barrier());
      before->push_back(before_b1);
    }
    before->append(_Declare(index_type, "y_tid", ((tid / 1) % 4)));
    auto y_tid = _("y_tid");
    before->append(_Declare(index_type, "x_tid", ((tid / 4) % 4)));
    auto x_tid = _("x_tid");
    before->append(_DeclareConst(index_type, "y_lid", 0));
    auto y_lid = _("y_lid");
    before->append(_Declare(index_type, "y", ((4 * y_lid) + y_tid)));
    auto y = _("y");
    before->append(_DeclareConst(index_type, "x_lid", 0));
    auto x_lid = _("x_lid");
    before->append(_Declare(index_type, "x", ((4 * x_lid) + x_tid)));
    auto x = _("x");
    before->append(_Declare(short_type, "LC", agg[((0 + (y_lid * 1)) + (x_lid * 1))]));
    auto LC = _("LC");
    before->append(_Declare(index_type, "gout_idx", ((0 + (4 * (x_gid + x))) + (1 * (y_gid + y)))));
    auto gout_idx = _("gout_idx");
    before->append(C[gout_idx] = LC);
  }

  // void kernel__0(short* C, const short* in1, const short* in2)
  // {
  //   int tid = get_local_id(0);
  //   short agg[1] = {0, };
  //   short in1_shared[16];
  //   short in2_shared[16];
  //   {
  //     {
  //       int y_x_tid = (tid % 16);
  //       in1_shared[y_x_tid] = in1[clamp(y_x_tid, 0, 15)];
  //     }
  //     {
  //       int x_y_tid = (tid % 16);
  //       in2_shared[x_y_tid] = in2[clamp(x_y_tid, 0, 15)];
  //     }
  //     barrier();
  //     int y_tid = (tid % 4);
  //     int x_tid = ((tid / 4) % 4);
  //     short val1 = ((short)in1_shared[(y_tid + (4 * x_tid))]);
  //     short val2 = ((short)in2_shared[(x_tid + (4 * y_tid))]);
  //     short pre_agg = ((short)(val2 * val1));
  //     agg[0] = (agg[0] + pre_agg);
  //     barrier();
  //   }
  //   int y_tid = (tid % 4);
  //   int x_tid = ((tid / 4) % 4);
  //   short LC = agg[0];
  //   int gout_idx = ((4 * x_tid) + y_tid);
  //   C[gout_idx] = LC;
  // }
  auto after = _Block({});
  {
    auto C = _("C");
    auto in1 = _("in1");
    auto in2 = _("in2");
    after->append(_Declare(index_type, "tid", _Index(sem::IndexExpr::LOCAL, 0)));
    auto tid = _("tid");
    after->append(_Declare({sem::Type::VALUE, lang::DataType::INT16, 1, 1}, "agg",
                           _LimitConst(sem::LimitConst::ZERO, lang::DataType::INT16)));
    auto agg = _("agg");
    after->append(_Declare({sem::Type::VALUE, lang::DataType::INT16, 1, 16}, "in1_shared", nullptr));
    auto in1_shared = _("in1_shared");
    after->append(_Declare({sem::Type::VALUE, lang::DataType::INT16, 1, 16}, "in2_shared", nullptr));
    auto in2_shared = _("in2_shared");
    {
      auto after_b1 = _Block({});
      {
        auto after_b1_b1 = _Block({});
        after_b1_b1->append(_Declare(index_type, "y_x_tid", (tid % 16)));
        auto y_x_tid = _("y_x_tid");
        after_b1_b1->append(in1_shared[y_x_tid] = in1[_Clamp(y_x_tid, _Const(0), _Const(15))]);
        after_b1->push_back(after_b1_b1);
      }
      {
        auto after_b1_b2 = _Block({});
        after_b1_b2->append(_Declare(index_type, "x_y_tid", (tid % 16)));
        auto x_y_tid = _("x_y_tid");
        after_b1_b2->append(in2_shared[x_y_tid] = in2[_Clamp(x_y_tid, _Const(0), _Const(15))]);
        after_b1->push_back(after_b1_b2);
      }
      after_b1->append(_Barrier());
      after_b1->append(_Declare(index_type, "y_tid", (tid % 4)));
      auto y_tid = _("y_tid");
      after_b1->append(_Declare(index_type, "x_tid", ((tid / 4) % 4)));
      auto x_tid = _("x_tid");
      after_b1->append(_Declare(short_type, "val1", _Cast(short_type, in1_shared[y_tid + (4 * x_tid)])));
      auto val1 = _("val1");
      after_b1->append(_Declare(short_type, "val2", _Cast(short_type, in2_shared[x_tid + (4 * y_tid)])));
      auto val2 = _("val2");
      after_b1->append(_Declare(short_type, "pre_agg", _Cast(short_type, (val2 * val1))));
      auto pre_agg = _("pre_agg");
      after_b1->append(agg[_Const(0)] = (agg[_Const(0)] + pre_agg));
      after_b1->append(_Barrier());
      after->push_back(after_b1);
    }
    after->append(_Declare(index_type, "y_tid", (tid % 4)));
    auto y_tid = _("y_tid");
    after->append(_Declare(index_type, "x_tid", ((tid / 4) % 4)));
    auto x_tid = _("x_tid");
    after->append(_Declare(short_type, "LC", agg[_Const(0)]));
    auto LC = _("LC");
    after->append(_Declare(index_type, "gout_idx", ((4 * x_tid) + y_tid)));
    auto gout_idx = _("gout_idx");
    after->append(C[gout_idx] = LC);
  }

  auto fn_before = _Function("kernel", sem::Type{}, {}, {before});
  auto fn_after = _Function("kernel", sem::Type{}, {}, {after});
  return TestParam{fn_before, fn_after};
}

INSTANTIATE_TEST_CASE_P(Samples, SimplifierTest, ::testing::Values(Basic(), Contraction()));

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
