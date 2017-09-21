
#include <gtest/gtest.h>
#include <stdio.h>

#include "base/util/astgen/test_ast.h"

using namespace vertexai::ast;  // NOLINT

namespace {

TEST(astgen_tests, smoke) {
  Node x = _BinaryOp("+", _IntegerConst(0), _BinaryOp("*", _IntegerConst(1), _SymLookup("foo")));
  Variable<Expression> e;
  Variable<std::string> op;
  Variable<int> ic;
  Variable<SymLookup> sl;
  Matcher plusZero = _BinaryOp("+", _IntegerConst(0), _BinaryOp(op, _IntegerConst(ic), sl));
  bool r = plusZero->Match(x);
  ASSERT_EQ(r, true);
  ASSERT_EQ(*ic, 1);
  ASSERT_EQ(sl->name(), "foo");
}

}  // namespace
