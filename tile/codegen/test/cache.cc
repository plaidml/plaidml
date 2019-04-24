// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "tile/codegen/cache.h"
#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lib/lib.h"
#include "tile/stripe/stripe.h"

using ::testing::ContainerEq;
using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

using namespace stripe;  // NOLINT

TEST(Codegen, Cache) {
  std::map<std::string, std::vector<float>> data = {
      {"A",
       {
           1, 2, 3, 4, 5,  //
           4, 5, 6, 7, 8,  //
           7, 8, 9, 7, 8,  //
           1, 2, 3, 1, 2,  //
           1, 2, 3, 1, 2,  //
       }},
      {"B",
       {
           1, 2, 3, 1, 2,  //
           1, 2, 3, 1, 2,  //
           1, 2, 3, 1, 2,  //
           1, 2, 3, 1, 2,  //
           1, 2, 3, 1, 2,  //
       }},
      {"C",
       {
           0, 0, 0, 0, 0,  //
           0, 0, 0, 0, 0,  //
           0, 0, 0, 0, 0,  //
           0, 0, 0, 0, 0,  //
           0, 0, 0, 0, 0,  //
       }},
  };

  std::vector<float> expected = {
      15, 30, 45,  15, 30,  //
      30, 60, 90,  30, 60,  //
      39, 78, 117, 39, 78,  //
      9,  18, 27,  9,  18,  //
      9,  18, 27,  9,  18,  //
  };

  size_t dim = sqrt(expected.size());
  auto runinfo = lib::LoadMatMul("matmul",                                    //
                                 SimpleShape(DataType::FLOAT32, {dim, dim}),  //
                                 SimpleShape(DataType::FLOAT32, {dim, dim}));
  auto program = GenerateStripe(runinfo);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);
  AliasMap am(AliasMap(AliasMap(AliasMap(), program->entry.get()), main.get()), kernel.get());
  IVLOG(2, "Original>\n" << *program->entry);

  ApplyTile(kernel.get(), {2, 2, 2});
  IVLOG(2, "Tiled>\n" << *program->entry);

  ApplyCache(am, kernel.get(), "A", {{{"CACHE"}}}, {{{"TX"}}});
  IVLOG(2, "Cached\n" << *program->entry);

  // ExecuteProgram(*program->entry, &data);

  // IVLOG(2, "A: " << data["A"]);
  // IVLOG(2, "B: " << data["B"]);
  // IVLOG(2, "C: " << data["C"]);
  // EXPECT_THAT(data["C"], ContainerEq(expected));
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
