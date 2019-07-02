// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "tile/codegen/cache.h"
#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/semprinter.h"
#include "tile/lib/lib.h"
#include "tile/ocl_exec/emitsem.h"
#include "tile/stripe/stripe.h"

using ::testing::ContainerEq;
using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

using namespace stripe;  // NOLINT
using plaidml::edsl::LogicalShape;

static std::vector<float> GenerateMatrix(size_t m, size_t n, bool zero) {
  std::vector<float> result(m * n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (zero) {
        result[i * m + j] = 0;
      } else {
        result[i * m + j] = std::rand();
      }
    }
  }
  return result;
}

TEST(Codegen, Unaligned) {
  int64_t dim = 112;
  std::map<std::string, std::vector<float>> data;
  data["A"] = test::GenerateMatrix(dim, dim, false);
  data["B"] = test::GenerateMatrix(dim, dim, false);
  data["C"] = test::GenerateMatrix(dim, dim, true);
  auto runinfo = lib::LoadMatMul("matmul",                                        //
                                 LogicalShape(PLAIDML_DATA_FLOAT32, {dim, dim}),  //
                                 LogicalShape(PLAIDML_DATA_FLOAT32, {dim, dim}));
  auto program = GenerateStripe(runinfo);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);
  AliasMap am(AliasMap(AliasMap(AliasMap(), program->entry.get()), main.get()), kernel.get());
  IVLOG(2, "Original>\n" << *program->entry);

  ApplyTile(kernel.get(), {32, 32, 32}, true, false, false, true);
  IVLOG(2, "Tiled>\n" << *program->entry);

  codegen::SemtreeEmitter emit(codegen::AliasMap{}, 256);
  emit.Visit(*program->entry);
  for (const auto ki : emit.kernels_.kernels) {
    sem::Print p(*ki.kfunc);
    IVLOG(2, p.str());
    IVLOG(2, "gids = " << ki.gwork);
    IVLOG(2, "lids = " << ki.lwork);
  }
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
