#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <tuple>

#include "base/util/logging.h"
#include "tile/lang/gid.h"

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Le;
using ::testing::Matches;
using ::testing::PrintToString;
using ::testing::SizeIs;

namespace vertexai {
namespace tile {
namespace lang {
namespace gid {

MATCHER_P(HasGid, gid, std::string("has gid_index=") + PrintToString(gid)) {
  *result_listener << "where gid_index=" << arg.gid_index;
  return Matches(gid)(arg.gid_index);
}

MATCHER(NoRightShift, "has no right_shift") {
  *result_listener << "where has_right_shift=" << arg.has_right_shift << " and right_shift=" << arg.right_shift;
  return !arg.has_right_shift;
}

MATCHER_P(HasRightShift, shift, std::string("has right_shift=") + PrintToString(shift)) {
  *result_listener << "where has_right_shift=" << arg.has_right_shift << " and right_shift=" << arg.right_shift;
  return arg.has_right_shift && Matches(shift)(arg.right_shift);
}

MATCHER(NoMask, "has no mask") {
  *result_listener << "where has_mask=" << arg.has_mask << " and mask=" << arg.mask;
  return !arg.has_mask;
}

MATCHER_P(HasMask, mask, std::string("has mask=") + PrintToString(mask)) {
  *result_listener << "where has_mask=" << arg.has_mask << " and mask=" << arg.mask;
  return arg.has_mask && Matches(mask)(arg.mask);
}

MATCHER(NoDivisor, "has no divisor") {
  *result_listener << "where has_divisor=" << arg.has_divisor << " and divisor=" << arg.divisor;
  return !arg.has_divisor;
}

MATCHER_P(HasDivisor, divisor, std::string("has divisor=") + PrintToString(divisor)) {
  *result_listener << "where has_divisor=" << arg.has_divisor << " and divisor=" << arg.divisor;
  return arg.has_divisor && Matches(divisor)(arg.divisor);
}

MATCHER(NoModulus, "has no modulus") {
  *result_listener << "where has_modulus=" << arg.has_modulus << " and modulus=" << arg.modulus;
  return !arg.has_modulus;
}

MATCHER_P(HasModulus, modulus, std::string("has modulus=") + PrintToString(modulus)) {
  *result_listener << "where has_modulus=" << arg.has_modulus << " and modulus=" << arg.modulus;
  return arg.has_modulus && Matches(modulus)(arg.modulus);
}

TEST(GidTest, OneGidOnePow2Dim) {
  auto map = MakeMap(std::vector<std::size_t>{32}, std::vector<std::size_t>{32});
  EXPECT_THAT(map.gid_sizes, ElementsAre(32));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, ThreeGidOnePow2Dim) {
  auto map = MakeMap(std::vector<std::size_t>{32, 32, 32}, std::vector<std::size_t>{32});
  EXPECT_THAT(map.gid_sizes, ElementsAre(32));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, OneGidThreePow2Dim) {
  auto map = MakeMap(std::vector<std::size_t>{64}, std::vector<std::size_t>{4, 4, 4});
  EXPECT_THAT(map.gid_sizes, ElementsAre(64));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(2), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(4), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, BigSmallGidThreePow2Dim) {
  auto map = MakeMap(std::vector<std::size_t>{64, 64}, std::vector<std::size_t>{4, 4, 4});
  EXPECT_THAT(map.gid_sizes, ElementsAre(64));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(2), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(4), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, OneSmallGidThreePow2Dim) {
  auto map = MakeMap(std::vector<std::size_t>{32}, std::vector<std::size_t>{4, 4, 4});
  EXPECT_THAT(map.gid_sizes, ElementsAre(64));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(2), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(4), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, SmallSmallGidThreePow2Dim) {
  auto map = MakeMap(std::vector<std::size_t>{32, 32}, std::vector<std::size_t>{4, 4, 4});
  EXPECT_THAT(map.gid_sizes, ElementsAre(16, 4));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(2), NoMask(), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(1), NoRightShift(), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, BigGidThreeDimOneMbit) {
  auto map = MakeMap(std::vector<std::size_t>{128, 32}, std::vector<std::size_t>{4, 4, 5});
  EXPECT_THAT(map.gid_sizes, ElementsAre(80));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(2), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(4), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, BigGidThreeDimTwoMbit) {
  auto map = MakeMap(std::vector<std::size_t>{128, 32}, std::vector<std::size_t>{3, 4, 5});
  EXPECT_THAT(map.gid_sizes, ElementsAre(12, 5));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), HasRightShift(2), NoMask(), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), NoRightShift(), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(1), NoRightShift(), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, SmallSmallGidThreeDimOneMbit) {
  auto map = MakeMap(std::vector<std::size_t>{32, 32}, std::vector<std::size_t>{4, 4, 5});
  EXPECT_THAT(map.gid_sizes, ElementsAre(20, 4));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(1), NoRightShift(), NoMask(), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(2), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, SmallSmallGidThreeDimTwoMbit) {
  auto map = MakeMap(std::vector<std::size_t>{8, 8}, std::vector<std::size_t>{3, 4, 5});
  EXPECT_THAT(map.gid_sizes, ElementsAre(12, 5));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), HasRightShift(2), NoMask(), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), NoRightShift(), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(1), NoRightShift(), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, SmallGidThreeDimTwoMbit) {
  auto map = MakeMap(std::vector<std::size_t>{8}, std::vector<std::size_t>{3, 4, 5});
  EXPECT_THAT(map.gid_sizes, ElementsAre(60));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), HasRightShift(2), NoMask(), NoDivisor(), HasModulus(3)),
                                    AllOf(HasGid(0), NoRightShift(), HasMask(3), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(2), NoMask(), HasDivisor(3), NoModulus())));
}

TEST(GidTest, SmallGidThreeDimThreeMbit) {
  auto map = MakeMap(std::vector<std::size_t>{2}, std::vector<std::size_t>{3, 5, 7});
  EXPECT_THAT(map.gid_sizes, ElementsAre(105));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), NoMask(), NoDivisor(), HasModulus(3)),
                                    AllOf(HasGid(0), NoRightShift(), NoMask(), HasDivisor(3), HasModulus(5)),
                                    AllOf(HasGid(0), NoRightShift(), NoMask(), HasDivisor(15), NoModulus())));
}

TEST(GidTest, ThreeSmallGidThreeDimThreeMbit) {
  auto map = MakeMap(std::vector<std::size_t>{2, 2, 2}, std::vector<std::size_t>{3, 5, 7});
  EXPECT_THAT(map.gid_sizes, ElementsAre(3, 5, 7));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), NoMask(), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(1), NoRightShift(), NoMask(), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(2), NoRightShift(), NoMask(), NoDivisor(), NoModulus())));
}

TEST(GidTest, UnevenDimensions) {
  // This was T980; these values were causing ComputeGrids to create
  // an lwork shape that was larger than the implementation supported.
  auto map = MakeMap(std::vector<std::size_t>{1024, 1024, 64}, std::vector<std::size_t>{16, 10, 10, 1536});
  EXPECT_THAT(map.gid_sizes, ElementsAre(160, 10, 1536));
  EXPECT_THAT(map.dims, ElementsAre(AllOf(HasGid(0), NoRightShift(), HasMask(15), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(0), HasRightShift(4), NoMask(), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(1), NoRightShift(), NoMask(), NoDivisor(), NoModulus()),
                                    AllOf(HasGid(2), NoRightShift(), NoMask(), NoDivisor(), NoModulus())));
  GridSize gwork;
  GridSize lwork;
  std::tie(gwork, lwork) = ComputeGrids(map, 128);
  EXPECT_THAT(lwork[0], Le(1024));
  EXPECT_THAT(lwork[1], Le(1024));
  EXPECT_THAT(lwork[2], Le(64));
}

}  // namespace gid
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
