// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "tile/codegen/alias.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

TEST(Codegen, AliasCheckOverlap) {
  EXPECT_TRUE(CheckOverlap({}, {}));
  EXPECT_TRUE(CheckOverlap(
      {
          {0, 10},
          {0, 10},
      },
      {
          {2, 4},
          {6, 8},
      }));
  EXPECT_TRUE(CheckOverlap(
      {
          {0, 10},
          {0, 10},
      },
      {
          {10, 20},
          {10, 20},
      }));
  EXPECT_FALSE(CheckOverlap(
      {
          {0, 10},
          {0, 10},
      },
      {
          {11, 20},
          {11, 20},
      }));
  EXPECT_FALSE(CheckOverlap(
      {
          {0, 10},
          {0, 10},
          {0, 0},
      },
      {
          {0, 10},
          {0, 10},
          {1, 1},
      }));
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
