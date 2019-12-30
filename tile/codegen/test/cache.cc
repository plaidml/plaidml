// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "plaidml2/edsl/helper.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/runinfo.h"
#include "tile/lib/lib.h"
#include "tile/stripe/stripe.h"

using ::testing::ContainerEq;
using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

using namespace stripe;  // NOLINT
using plaidml::edsl::LogicalShape;

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

  int64_t dim = sqrt(expected.size());
  auto tileProgram = lib::LoadMatMul(                  //
      "matmul",                                        //
      LogicalShape(PLAIDML_DATA_FLOAT32, {dim, dim}),  //
      LogicalShape(PLAIDML_DATA_FLOAT32, {dim, dim}));
  auto program = plaidml::edsl::ConvertIntoStripe(tileProgram);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);
  IVLOG(2, "Original>\n" << *program->entry);

  ApplyTile(kernel.get(), {2, 2, 2});
  IVLOG(2, "Tiled>\n" << *program->entry);

  auto inner = kernel.get()->SubBlock(0);
  // Must explicitly define program_map and main_map.
  // Otherwise, they may be freed in ApplyCache.
  AliasMap program_map(AliasMap(), program->entry.get());
  AliasMap main_map(program_map, main.get());
  AliasMap am(main_map, kernel.get());
  ApplyCache(am, RefDir::In, inner.get(), kernel.get(), "A", {{{"CACHE"}}}, {{{"TX"}}});
  IVLOG(2, "Cached\n" << *program->entry);

  // ExecuteProgram(*program->entry, &data);

  // IVLOG(2, "A: " << data["A"]);
  // IVLOG(2, "B: " << data["B"]);
  // IVLOG(2, "C: " << data["C"]);
  // EXPECT_THAT(data["C"], ContainerEq(expected));
}

TEST(Codegen, CacheConv2d) {
  auto tileProgram = lib::LoadConv2d(                        //
      "conv2d",                                              //
      LogicalShape(PLAIDML_DATA_FLOAT32, {100, 14, 14, 3}),  //
      LogicalShape(PLAIDML_DATA_FLOAT32, {3, 3, 3, 3}),      //
      {100, 12, 12, 3});
  auto program = plaidml::edsl::ConvertIntoStripe(tileProgram);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);

  // Must explicitly define program_map and main_map.
  // Otherwise, they may be freed in ApplyCache.
  AliasMap program_map(AliasMap(), program->entry.get());
  AliasMap main_map(program_map, main.get());
  IVLOG(2, "Original>\n" << *program->entry);

  ApplyTile(kernel.get(), {3, 3, 3, 3, 20, 6, 6});
  IVLOG(2, "Tiled>\n" << *program->entry);

  auto inner = kernel.get()->SubBlock(0);
  AliasMap am0(main_map, kernel.get());
  ApplyCache(am0, RefDir::In, inner.get(), kernel.get(), "I", {{{"CACHE"}}}, {{{"TX"}}});
  IVLOG(2, "Input cached\n" << *program->entry);

  inner = kernel.get()->SubBlock(1);
  AliasMap am1(main_map, kernel.get());
  ApplyCache(am1, RefDir::Out, inner.get(), kernel.get(), "O", {{{"CACHE"}}}, {{{"TX"}}});
  IVLOG(2, "Output cached\n" << *program->entry);
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
