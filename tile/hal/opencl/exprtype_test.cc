// Copyright 2017, Vertex.AI.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tile/hal/opencl/exprtype.h"
#include "tile/lang/sembuilder.h"

using ::testing::Eq;
using ::testing::Values;

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {
namespace {

using namespace sem::builder;  // NOLINT

MATCHER_P(IsType, ty, to_string(ty)) {
  return (arg.base == ty.base && arg.dtype == ty.dtype && arg.vec_width == ty.vec_width && arg.array == ty.array &&
          arg.region == ty.region);
}

MATCHER_P(IsValueType, dtype, "is value type with dtype=" + to_string(dtype)) {
  return arg.base == sem::Type::VALUE && arg.dtype == dtype;
}

class ExprTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scope_.Bind("idx", sem::Type{sem::Type::INDEX});
    scope_.Bind("val_int8", sem::Type{sem::Type::VALUE, lang::DataType::INT8});
  }

  sem::Type TypeOf(const sem::ExprPtr& expr) { return ExprType::TypeOf(&scope_, cl_khr_fp16_, expr); }

  bool cl_khr_fp16_ = false;
  lang::Scope<sem::Type> scope_;
};

TEST_F(ExprTypeTest, Constants) {
  EXPECT_THAT(TypeOf(_Const(0)), IsValueType(lang::DataType::INT8));

  EXPECT_THAT(TypeOf(_Const(-1)), IsValueType(lang::DataType::INT8));
  EXPECT_THAT(TypeOf(_Const(-128)), IsValueType(lang::DataType::INT8));
  EXPECT_THAT(TypeOf(_Const(-129)), IsValueType(lang::DataType::INT16));
  EXPECT_THAT(TypeOf(_Const(-32768)), IsValueType(lang::DataType::INT16));
  EXPECT_THAT(TypeOf(_Const(-32769)), IsValueType(lang::DataType::INT32));
  EXPECT_THAT(TypeOf(_Const(-2147483648)), IsValueType(lang::DataType::INT32));
  EXPECT_THAT(TypeOf(_Const(-2147483649)), IsValueType(lang::DataType::INT64));

  EXPECT_THAT(TypeOf(_Const(1)), IsValueType(lang::DataType::INT8));
  EXPECT_THAT(TypeOf(_Const(127)), IsValueType(lang::DataType::INT8));
  EXPECT_THAT(TypeOf(_Const(128)), IsValueType(lang::DataType::INT16));
  EXPECT_THAT(TypeOf(_Const(32767)), IsValueType(lang::DataType::INT16));
  EXPECT_THAT(TypeOf(_Const(32768)), IsValueType(lang::DataType::INT32));
  EXPECT_THAT(TypeOf(_Const(2147483647)), IsValueType(lang::DataType::INT32));
  EXPECT_THAT(TypeOf(_Const(2147483648)), IsValueType(lang::DataType::INT64));
}

TEST_F(ExprTypeTest, AddIndexes) { EXPECT_THAT(TypeOf(_("idx") + _("idx")), IsType(sem::Type{sem::Type::INDEX})); }

TEST_F(ExprTypeTest, IndexAddSmallConst) { EXPECT_THAT(TypeOf(_("idx") + 0), IsType(sem::Type{sem::Type::INDEX})); }

TEST_F(ExprTypeTest, SmallConstAddIndex) {
  EXPECT_THAT(TypeOf(_Const(0) + _("idx")), IsType(sem::Type{sem::Type::INDEX}));
}

TEST_F(ExprTypeTest, IndexAddBigConst) {
  EXPECT_THAT(TypeOf(_("idx") + 2147483648), IsType(sem::Type{sem::Type::VALUE, lang::DataType::INT64}));
}

TEST_F(ExprTypeTest, BigConstAddIndex) {
  EXPECT_THAT(TypeOf(_Const(2147483648) + _("idx")), IsType(sem::Type{sem::Type::VALUE, lang::DataType::INT64}));
}

TEST_F(ExprTypeTest, IndexCompareSmallConst) {
  EXPECT_THAT(TypeOf(_("idx") <= 8), IsType(sem::Type{sem::Type::VALUE, lang::DataType::INT32}));
}

TEST_F(ExprTypeTest, SmallConstCompareIndex) {
  EXPECT_THAT(TypeOf(_Const(8) <= _("idx")), IsType(sem::Type{sem::Type::VALUE, lang::DataType::INT32}));
}

}  // namespace
}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
