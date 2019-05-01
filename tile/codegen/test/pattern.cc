// Copyright 2019, Intel Corp.

#include <gmock/gmock.h>

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

using namespace lang;    // NOLINT
using namespace stripe;  // NOLINT

void RoundTrip(const std::string& code) { EXPECT_THAT(to_string(Parse(code)), Eq(code)); }

TEST(Pattern, Lexer) {
  EXPECT_THAT(GetTokens("foo()."), ContainerEq(std::vector<std::string>{"foo", "(", ")", "."}));
  EXPECT_THAT(GetTokens(" \t\nfoo(   )   .   "), ContainerEq(std::vector<std::string>{"foo", "(", ")", ".", ""}));
  EXPECT_THAT(GetTokens("foo(bar, baz)."),
              ContainerEq(std::vector<std::string>{"foo", "(", "bar", ",", "baz", ")", "."}));
}

TEST(Pattern, Parser) {
  RoundTrip("foo()");
  RoundTrip("foo(bar)");
  RoundTrip("foo(bar, baz)");
  RoundTrip("foo(bar(baz))");
  RoundTrip("foo(bar(baz, buck))");
  RoundTrip("foo([bar, baz])");
  RoundTrip("block(ref(in, [dim(_, X, _), dim(_, _, _), dim(_, _, _), dim(_, _, 1)]))");
}

TEST(Pattern, Match) {
  auto value = Parse("top([ a([ 1, 50 ]), a([ 5, 50 ]) ])");
  auto pattern = Parse("top({ a({ X, Y }), a({ Y, Z }) })");
  auto result = Match(pattern, value);
  EXPECT_TRUE(result.matched);
  EXPECT_THAT(to_string(result.vars["X"]), Eq("1"));
  EXPECT_THAT(to_string(result.vars["Y"]), Eq("50"));
  EXPECT_THAT(to_string(result.vars["Z"]), Eq("5"));
}

Term GeneratePattern() {
  return Parse(R"(
block({
  ref(out, {
    dim(0, {term(1, X1)}, _, _),
    dim(0, {term(1, CO)}, _, 1),
    dim(0, {term(1, X0)}, _, _),
    dim(0, {term(1, N)},  _, _)
  }),
  ref(in, {
    dim(P0, {term(1, K0), term(S0, X0)}, _, _),
    dim(0, {term(1, N)},  _, _),
    dim(P1, {term(1, K1), term(S1, X1)}, _, _),
    dim(0, {term(1, CI)}, _, 1)
  }),
  ref(in, {
    dim(0, {term(1, K1)}, _, _),
    dim(0, {term(1, CO)}, _, 1),
    dim(0, {term(1, CI)}, _, _),
    dim(0, {term(1, K0)}, _, _)
  })
})
  )");
}

TEST(Pattern, Conv1x1s1) {
  Tensor I(SimpleShape(DataType::INT8, {1, 100, 100, 56}), "I");
  Tensor K(SimpleShape(DataType::INT8, {1, 1, 56, 56}), "K");
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
])
)");
  EXPECT_THAT(to_string(term), Eq(to_string(expected)));
  auto pattern = GeneratePattern();
  IVLOG(1, pattern);
  auto result = Match(pattern, term);
  EXPECT_TRUE(result.matched);
  EXPECT_THAT(to_string(result.vars["P0"]), Eq("0"));
  EXPECT_THAT(to_string(result.vars["P1"]), Eq("0"));
  EXPECT_THAT(to_string(result.vars["S0"]), Eq("1"));
  EXPECT_THAT(to_string(result.vars["S1"]), Eq("1"));
}

TEST(Pattern, Conv3x3s1) {
  Tensor I(SimpleShape(DataType::INT8, {1, 100, 100, 56}), "I");
  Tensor K(SimpleShape(DataType::INT8, {3, 3, 56, 56}), "K");
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
])
)");
  EXPECT_THAT(to_string(term), Eq(to_string(expected)));
  auto pattern = GeneratePattern();
  IVLOG(1, pattern);
  auto result = Match(pattern, term);
  EXPECT_TRUE(result.matched);
  EXPECT_THAT(to_string(result.vars["P0"]), Eq("-1"));
  EXPECT_THAT(to_string(result.vars["P1"]), Eq("-1"));
  EXPECT_THAT(to_string(result.vars["S0"]), Eq("1"));
  EXPECT_THAT(to_string(result.vars["S1"]), Eq("1"));
}

TEST(Pattern, Conv7x7s2) {
  Tensor I(SimpleShape(DataType::INT8, {1, 224, 224, 3}), "I");
  Tensor K(SimpleShape(DataType::INT8, {7, 7, 3, 64}), "K");
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
])
)");
  EXPECT_THAT(to_string(term), Eq(to_string(expected)));
  auto pattern = GeneratePattern();
  IVLOG(1, pattern);
  auto result = Match(pattern, term);
  EXPECT_TRUE(result.matched);
  EXPECT_THAT(to_string(result.vars["P0"]), Eq("-3"));
  EXPECT_THAT(to_string(result.vars["P1"]), Eq("-3"));
  EXPECT_THAT(to_string(result.vars["S0"]), Eq("2"));
  EXPECT_THAT(to_string(result.vars["S1"]), Eq("2"));
}

}  // namespace test
}  // namespace pattern
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
