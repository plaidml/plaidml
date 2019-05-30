// Copyright 2019, Intel Corp.

#include <gmock/gmock.h>

#include <vector>

#include "tile/codegen/pattern.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lib/lib.h"
#include "tile/stripe/stripe.h"

using ::testing::ContainerEq;
using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace pattern {
namespace test {

using namespace lang;           // NOLINT
using namespace stripe;         // NOLINT
using namespace plaidml::edsl;  // NOLINT

void RoundTrip(const std::string& code) { EXPECT_THAT(to_string(Parse(code)), Eq(code)); }

TEST(Pattern, Lexer) {
  EXPECT_THAT(GetTokens("foo()."), ContainerEq(std::vector<std::string>{"foo", "(", ")", "."}));
  EXPECT_THAT(GetTokens(" \t\nfoo(   )   .   "), ContainerEq(std::vector<std::string>{"foo", "(", ")", "."}));
  EXPECT_THAT(                      //
      GetTokens("foo(bar, baz)."),  //
      ContainerEq(std::vector<std::string>{"foo", "(", "bar", ",", "baz", ")", "."}));
  EXPECT_THAT(                                   //
      GetTokens("foo(_bar, X_baz, x_y, -15)."),  //
      ContainerEq(std::vector<std::string>{"foo", "(", "_bar", ",", "X_baz", ",", "x_y", ",", "-15", ")", "."}));
  EXPECT_THAT(GetTokens(""), ContainerEq(std::vector<std::string>{}));
  EXPECT_THROW(GetTokens("foo(@_bar, X_baz, x_y, -15)."), std::runtime_error);
  EXPECT_THROW(GetTokens("foo(_bar, X_baz, x_y, -15)'."), std::runtime_error);
}

TEST(Pattern, Parser) {
  RoundTrip("foo()");
  RoundTrip("foo(bar)");
  RoundTrip("foo(bar, baz)");
  RoundTrip("foo(bar(baz))");
  RoundTrip("foo(bar(baz, buck))");
  RoundTrip("foo([bar, baz])");
  RoundTrip("block(ref(in, [dim(_, X, _), dim(_, _, _), dim(_, _, _), dim(_, _, 1)]))");
  RoundTrip("foo(-15, 15, _, _x, x_x, X, X_x, x1, X1, _x1, _X1)");
}

TEST(Pattern, MatchFirst) {
  auto value = Parse("top([ a([ 1, 50 ]), a([ 5, 50 ]) ])");
  auto pattern = Parse("top({ a({ X, Y }), a({ Y, Z }) })");
  auto result = MatchFirst(pattern, value);
  ASSERT_TRUE(result);
  EXPECT_THAT(to_string(*result), Eq("{(X->1), (Y->50), (Z->5)}"));
}

TEST(Pattern, MatchNone) {
  EXPECT_THAT(                             //
      to_string(MatchAll(Parse("a(x)"),    //
                         Parse("a(y)"))),  //
      ContainerEq(std::vector<std::string>{}));
}

TEST(Pattern, MatchAll1) {
  EXPECT_THAT(                               //
      to_string(MatchAll(Parse("a({X})"),    //
                         Parse("a([1])"))),  //
      ContainerEq(std::vector<std::string>{
          "{(X->1)}",
      }));
  EXPECT_THAT(                                  //
      to_string(MatchAll(Parse("a(P, {X})"),    //
                         Parse("a(0, [1])"))),  //
      ContainerEq(std::vector<std::string>{
          "{(P->0), (X->1)}",
      }));
  EXPECT_THAT(                                         //
      to_string(MatchAll(Parse("a({ b(P, {X}) })"),    //
                         Parse("a([ b(0, [1]) ])"))),  //
      ContainerEq(std::vector<std::string>{
          "{(P->0), (X->1)}",
      }));
  EXPECT_THAT(                                     //
      to_string(MatchAll(                          //
          Parse("a({ b(0, {_}), b(P, {X}) })"),    //
          Parse("a([ b(0, [1]), b(0, [1]) ])"))),  //
      ContainerEq(std::vector<std::string>{
          "{(P->0), (X->1)}",
      }));
}

TEST(Pattern, MatchAll2) {
  EXPECT_THAT(                   //
      to_string(MatchAll(        //
          Parse("a({X, Y})"),    //
          Parse("a([1, 2])"))),  //
      ContainerEq(std::vector<std::string>{
          "{(X->1), (Y->2)}",
          "{(X->2), (Y->1)}",
      }));
  EXPECT_THAT(                           //
      to_string(MatchAll(                //
          Parse("a({X, Y}, {X, Y})"),    //
          Parse("a([1, 2], [1, 2])"))),  //
      ContainerEq(std::vector<std::string>{
          "{(X->1), (Y->2)}",
          "{(X->2), (Y->1)}",
      }));
  EXPECT_THAT(                                     //
      to_string(MatchAll(                          //
          Parse("a({ b({X, Y}), b({X, Y}) })"),    //
          Parse("a([ b([1, 2]), b([1, 2]) ])"))),  //
      ContainerEq(std::vector<std::string>{
          "{(X->1), (Y->2)}",
          "{(X->2), (Y->1)}",
      }));
  EXPECT_THAT(                                                                              //
      to_string(MatchAll(                                                                   //
          Parse("blk({ ref(in, { dim(0, { t(_) }), dim(P, { t(X) }) }) }, { idx(X) })"),    //
          Parse("blk([ ref(in, [ dim(0, [ t(0) ]), dim(0, [ t(x) ]) ]) ], [ idx(x) ])"))),  //
      ContainerEq(std::vector<std::string>{
          "{(P->0), (X->x)}",
      }));
}

TEST(Pattern, MatchAll3) {
  EXPECT_THAT(                      //
      to_string(MatchAll(           //
          Parse("a({X, Y, Z})"),    //
          Parse("a([1, 2, 3])"))),  //
      ContainerEq(std::vector<std::string>{
          "{(X->1), (Y->2), (Z->3)}",
          "{(X->1), (Y->3), (Z->2)}",
          "{(X->2), (Y->1), (Z->3)}",
          "{(X->2), (Y->3), (Z->1)}",
          "{(X->3), (Y->1), (Z->2)}",
          "{(X->3), (Y->2), (Z->1)}",
      }));
  EXPECT_THAT(                                             //
      to_string(MatchAll(                                  //
          Parse("top({ a({ X, Y }),  a({ Y, Z }) })"),     //
          Parse("top([ a([ 1, 50 ]), a([ 5, 50 ]) ])"))),  //
      ContainerEq(std::vector<std::string>{
          "{(X->1), (Y->50), (Z->5)}",
          "{(X->5), (Y->50), (Z->1)}",
      }));
}

Term GeneratePattern() {
  return Parse(R"(
block({
  ref(out, {
    dim(0, {term(1, CO)}, _, 1),
    dim(0, {term(1, X0)}, _, _),
    dim(0, {term(1, N)},  _, _),
    dim(0, {term(1, X1)}, _, _)
  }),
  ref(in, {
    dim(0,  {term(1, CI)}, _, 1),
    dim(P1, {term(1, K1), term(S1, X1)}, _, _),
    dim(0,  {term(1, N)}, _, _),
    dim(P0, {term(1, K0), term(S0, X0)}, _, _)
  }),
  ref(in, {
    dim(0, {term(1, CO)}, _, 1),
    dim(0, {term(1, CI)}, _, _),
    dim(0, {term(1, K0)}, _, _),
    dim(0, {term(1, K1)}, _, _)
  })
}, {
  idx(N, N_range),
  idx(X0, X0_range),
  idx(X1, X1_range),
  idx(K0, K0_range),
  idx(K1, K1_range),
  idx(CI, CI_range),
  idx(CO, CO_range)
})
  )");
}

TEST(Pattern, MatchBlock) {
  auto result = to_string(MatchAll(                         //
      Parse("ref({term(1, CI)}, dim(P0, {term(1, K0)}))"),  //
      Parse("ref([term(1, ci)], dim(0 , [term(1, k0)]))")));
  EXPECT_THAT(  //
      result,   //
      ContainerEq(std::vector<std::string>{
          "{(CI->ci), (K0->k0), (P0->0)}",
      }));
}

TEST(Pattern, Conv1x1s1) {
  Tensor I("I", SimpleShape(DataType::INT8, {1, 100, 100, 56}));
  Tensor K("K", SimpleShape(DataType::INT8, {1, 1, 56, 56}));
  std::vector<size_t> O_dims = {1, 100, 100, 56};
  auto runinfo = Evaluate("conv1x1s1", {lib::Convolution(I, K, O_dims)});
  runinfo.const_inputs = {"K"};
  auto program = GenerateStripe(runinfo);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);
  IVLOG(2, *main);
  auto term = IntoTerm(*kernel);
  IVLOG(1, term);
  auto expected = Parse(R"(
block([
  ref(in, [
    dim(0, [term(1, n)], 1, 560000),
    dim(0, [term(1, k0), term(1, x0)], 1, 5600),
    dim(0, [term(1, k1), term(1, x1)], 1, 56),
    dim(0, [term(1, ci)], 1, 1)
  ]),
  ref(in, [
    dim(0, [term(1, k0)], 1, 3136),
    dim(0, [term(1, k1)], 1, 3136),
    dim(0, [term(1, ci)], 1, 56),
    dim(0, [term(1, co)], 1, 1)
  ]),
  ref(out, [
    dim(0, [term(1, n)], 1, 560000),
    dim(0, [term(1, x0)], 1, 5600),
    dim(0, [term(1, x1)], 1, 56),
    dim(0, [term(1, co)], 1, 1)
  ])
], [
  idx(ci, 56),
  idx(co, 56),
  idx(k0, 1),
  idx(k1, 1),
  idx(n, 1),
  idx(x0, 100),
  idx(x1, 100)
])
)");
  EXPECT_THAT(to_string(term), Eq(to_string(expected)));
  auto pattern = GeneratePattern();
  IVLOG(1, pattern);
  EXPECT_THAT(to_string(MatchAll(pattern, term)),  //
              ContainerEq(std::vector<std::string>{
                  "{(CI->ci), (CI_range->56), (CO->co), (CO_range->56), (K0->k0), (K0_range->1), (K1->k1), "
                  "(K1_range->1), (N->n), (N_range->1), (P0->0), (P1->0), (S0->1), (S1->1), (X0->x0), (X0_range->100), "
                  "(X1->x1), (X1_range->100)}",
                  "{(CI->ci), (CI_range->56), (CO->co), (CO_range->56), (K0->k1), (K0_range->1), (K1->k0), "
                  "(K1_range->1), (N->n), (N_range->1), (P0->0), (P1->0), (S0->1), (S1->1), (X0->x1), (X0_range->100), "
                  "(X1->x0), (X1_range->100)}",
              }));
}

TEST(Pattern, Conv3x3s1) {
  Tensor I("I", SimpleShape(DataType::INT8, {1, 100, 100, 56}));
  Tensor K("K", SimpleShape(DataType::INT8, {3, 3, 56, 56}));
  std::vector<size_t> O_dims = {1, 100, 100, 56};
  auto runinfo = Evaluate("conv3x3s1", {lib::Convolution(I, K, O_dims)});
  runinfo.const_inputs = {"K"};
  auto program = GenerateStripe(runinfo);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);
  IVLOG(2, *main);
  auto term = IntoTerm(*kernel);
  auto expected = Parse(R"(
block([
  ref(in, [
    dim(0, [term(1, n)], 1, 560000),
    dim(-1, [term(1, k0), term(1, x0)], 1, 5600),
    dim(-1, [term(1, k1), term(1, x1)], 1, 56),
    dim(0, [term(1, ci)], 1, 1)
  ]),
  ref(in, [
    dim(0, [term(1, k0)], 1, 9408),
    dim(0, [term(1, k1)], 1, 3136),
    dim(0, [term(1, ci)], 1, 56),
    dim(0, [term(1, co)], 1, 1)
  ]),
  ref(out, [
    dim(0, [term(1, n)], 1, 560000),
    dim(0, [term(1, x0)], 1, 5600),
    dim(0, [term(1, x1)], 1, 56),
    dim(0, [term(1, co)], 1, 1)
  ])
], [
  idx(ci, 56),
  idx(co, 56),
  idx(k0, 3),
  idx(k1, 3),
  idx(n, 1),
  idx(x0, 100),
  idx(x1, 100)
])
)");
  EXPECT_THAT(to_string(term), Eq(to_string(expected)));
  auto pattern = GeneratePattern();
  IVLOG(1, pattern);
  EXPECT_THAT(
      to_string(MatchAll(pattern, term)),  //
      ContainerEq(std::vector<std::string>{
          "{(CI->ci), (CI_range->56), (CO->co), (CO_range->56), (K0->k0), (K0_range->3), (K1->k1), "
          "(K1_range->3), (N->n), (N_range->1), (P0->-1), (P1->-1), (S0->1), (S1->1), (X0->x0), (X0_range->100), "
          "(X1->x1), (X1_range->100)}",
          "{(CI->ci), (CI_range->56), (CO->co), (CO_range->56), (K0->k1), (K0_range->3), (K1->k0), "
          "(K1_range->3), (N->n), (N_range->1), (P0->-1), (P1->-1), (S0->1), (S1->1), (X0->x1), (X0_range->100), "
          "(X1->x0), (X1_range->100)}",
      }));
}

TEST(Pattern, Conv7x7s2) {
  Tensor I("I", SimpleShape(DataType::INT8, {1, 224, 224, 3}));
  Tensor K("K", SimpleShape(DataType::INT8, {7, 7, 3, 64}));
  std::vector<size_t> O_dims = {1, 112, 112, 64};
  auto runinfo = Evaluate("conv7x7s2", {lib::Convolution(I, K, O_dims, {2, 2})});
  runinfo.const_inputs = {"K"};
  auto program = GenerateStripe(runinfo);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);
  IVLOG(2, *main);
  auto term = IntoTerm(*kernel);
  auto expected = Parse(R"(
block([
  ref(in, [
    dim(0, [term(1, n)], 1, 150528),
    dim(-3, [term(1, k0), term(2, x0)], 1, 672),
    dim(-3, [ term(1, k1), term(2, x1)], 1, 3),
    dim(0, [term(1, ci)], 1, 1)
  ]),
  ref(in, [
    dim(0, [term(1, k0)], 1, 1344),
    dim(0, [term(1, k1)], 1, 192),
    dim(0, [term(1, ci)], 1, 64),
    dim(0, [term(1, co)], 1, 1)
  ]),
  ref(out, [
    dim(0, [term(1, n)], 1, 802816),
    dim(0, [term(1, x0)], 1, 7168),
    dim(0, [term(1, x1)], 1, 64),
    dim(0, [term(1, co)], 1, 1)
  ])
], [
  idx(ci, 3),
  idx(co, 64),
  idx(k0, 7),
  idx(k1, 7),
  idx(n, 1),
  idx(x0, 112),
  idx(x1, 112)
])
)");
  EXPECT_THAT(to_string(term), Eq(to_string(expected)));
  auto pattern = GeneratePattern();
  IVLOG(1, pattern);
  EXPECT_THAT(
      to_string(MatchAll(pattern, term)),  //
      ContainerEq(std::vector<std::string>{
          "{(CI->ci), (CI_range->3), (CO->co), (CO_range->64), (K0->k0), (K0_range->7), (K1->k1), "
          "(K1_range->7), (N->n), (N_range->1), (P0->-3), (P1->-3), (S0->2), (S1->2), (X0->x0), (X0_range->112), "
          "(X1->x1), (X1_range->112)}",
          "{(CI->ci), (CI_range->3), (CO->co), (CO_range->64), (K0->k1), (K0_range->7), (K1->k0), "
          "(K1_range->7), (N->n), (N_range->1), (P0->-3), (P1->-3), (S0->2), (S1->2), (X0->x1), (X0_range->112), "
          "(X1->x0), (X1_range->112)}",
      }));
}

}  // namespace test
}  // namespace pattern
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
