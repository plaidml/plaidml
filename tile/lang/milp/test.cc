#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tile/lang/bignum.h"
#include "tile/lang/milp/ilp_solver.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace milp {

TEST(MilpTest, TestTest) { EXPECT_EQ(0, 0); }

TEST(MilpTest, RationalTest) { EXPECT_EQ(Rational(0, 1), 0); }

TEST(MilpTest, BasicTableauTest) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(Polynomial("x") + 4, 4);
  Polynomial obj = Polynomial("x", 3);
  ILPSolver solver;
  Tableau t = solver.makeStandardFormTableau(constraints, obj);

  EXPECT_EQ(t.mat().size1(), 3);
  EXPECT_EQ(t.mat().size2(), 6);

  EXPECT_EQ(t.mat()(0, 0), 1);
  EXPECT_EQ(t.mat()(0, 1), -3);
  EXPECT_EQ(t.mat()(0, 2), 3);
  EXPECT_EQ(t.mat()(0, 3), 0);
  EXPECT_EQ(t.mat()(0, 4), 0);
  EXPECT_EQ(t.mat()(0, 5), 0);

  EXPECT_EQ(t.mat()(1, 0), 0);
  EXPECT_EQ(t.mat()(1, 1), -1);
  EXPECT_EQ(t.mat()(1, 2), 1);
  EXPECT_EQ(t.mat()(1, 3), 1);
  EXPECT_EQ(t.mat()(1, 4), 0);
  EXPECT_EQ(t.mat()(1, 5), 4);

  EXPECT_EQ(t.mat()(2, 0), 0);
  EXPECT_EQ(t.mat()(2, 1), -1);
  EXPECT_EQ(t.mat()(2, 2), 1);
  EXPECT_EQ(t.mat()(2, 3), 0);
  EXPECT_EQ(t.mat()(2, 4), -1);
  EXPECT_EQ(t.mat()(2, 5), 1);
}

TEST(MilpTest, OptimizeCanonicalTest) {
  std::vector<std::string> blank_var_names;
  Tableau t(3, 6, blank_var_names);
  t.mat()(0, 0) = 1;
  t.mat()(0, 1) = 5;
  t.mat()(0, 2) = -5;
  t.mat()(1, 1) = -2;
  t.mat()(1, 2) = 4;
  t.mat()(1, 3) = 1;
  t.mat()(1, 5) = 2;
  t.mat()(2, 1) = 9;
  t.mat()(2, 2) = 3;
  t.mat()(2, 4) = 1;
  t.mat()(2, 5) = 7;

  t.selectBasicVars();
  EXPECT_EQ(t.makeCanonicalFormOptimal(), true);

  EXPECT_EQ(t.mat()(0, 0), 1);
  EXPECT_EQ(t.mat()(0, 1), 0);
  EXPECT_EQ(t.mat()(0, 2), Rational(-20, 3));
  EXPECT_EQ(t.mat()(0, 3), 0);
  EXPECT_EQ(t.mat()(0, 4), Rational(-5, 9));
  EXPECT_EQ(t.mat()(0, 5), Rational(-35, 9));

  EXPECT_EQ(t.mat()(1, 0), 0);
  EXPECT_EQ(t.mat()(1, 1), 0);
  EXPECT_EQ(t.mat()(1, 2), Rational(14, 3));
  EXPECT_EQ(t.mat()(1, 3), 1);
  EXPECT_EQ(t.mat()(1, 4), Rational(2, 9));
  EXPECT_EQ(t.mat()(1, 5), Rational(32, 9));

  EXPECT_EQ(t.mat()(2, 0), 0);
  EXPECT_EQ(t.mat()(2, 1), 1);
  EXPECT_EQ(t.mat()(2, 2), Rational(1, 3));
  EXPECT_EQ(t.mat()(2, 3), 0);
  EXPECT_EQ(t.mat()(2, 4), Rational(1, 9));
  EXPECT_EQ(t.mat()(2, 5), Rational(7, 9));
}

TEST(MilpTest, SimpleOptimizeTest) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(Polynomial("x") + 4, 4);
  Polynomial obj = 3 * Polynomial("x");
  ILPSolver solver;
  Tableau t = solver.makeStandardFormTableau(constraints, obj);
  EXPECT_EQ(t.makeOptimal(), true);

  std::map<std::string, Rational> soln = t.reportSolution();

  EXPECT_EQ(soln["x_pos"], 0);
  EXPECT_EQ(soln["x_neg"], 4);
}

TEST(MilpTest, OptimizeTest2D) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(Polynomial("x") + Polynomial("y") + 2, 4);
  constraints.emplace_back(Polynomial("x") + 1, 4);
  constraints.emplace_back(Polynomial("y") + 2, 5);
  Polynomial obj = -3 * Polynomial("x") + 2 * Polynomial("y");
  ILPSolver solver;
  Tableau t = solver.makeStandardFormTableau(constraints, obj);
  EXPECT_EQ(t.makeOptimal(), true);

  std::map<std::string, Rational> soln = t.reportSolution();

  EXPECT_EQ(soln["x_pos"], 2);
  EXPECT_EQ(soln["x_neg"], 0);
  EXPECT_EQ(soln["y_pos"], 0);
  EXPECT_EQ(soln["y_neg"], 2);
}

TEST(MilpTest, TrivialILPTest) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(Polynomial("x") + Polynomial("y") + 2, 4);
  constraints.emplace_back(Polynomial("x") + 1, 4);
  constraints.emplace_back(Polynomial("y") + 2, 5);
  Polynomial obj = -3 * Polynomial("x") + 2 * Polynomial("y");
  ILPSolver solver;
  Tableau t = solver.makeStandardFormTableau(constraints, obj);
  EXPECT_TRUE(solver.solve(t));

  std::map<std::string, Rational> soln = solver.reportSolution();

  EXPECT_EQ(soln["x_pos"], 2);
  EXPECT_EQ(soln["x_neg"], 0);
  EXPECT_EQ(soln["y_pos"], 0);
  EXPECT_EQ(soln["y_neg"], 2);
}

TEST(MilpTest, ILPTest2D) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(2 * Polynomial("x") + Polynomial("y") + 2, 6);
  constraints.emplace_back(Polynomial("x") + 1, 4);
  constraints.emplace_back(Polynomial("y") + 2, 5);
  Polynomial obj = -3 * Polynomial("x") + 2 * Polynomial("y");
  ILPSolver solver;
  EXPECT_TRUE(solver.solve(constraints, obj));

  std::map<std::string, Rational> soln = solver.reportSolution();

  EXPECT_EQ(soln["x_pos"], 2);
  EXPECT_EQ(soln["x_neg"], 0);
  EXPECT_EQ(soln["y_pos"], 0);
  EXPECT_EQ(soln["y_neg"], 2);

  EXPECT_EQ(solver.reportObjective(), -10);
}

TEST(MilpTest, Subdivision1D) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(Polynomial("i_0"), 2);
  constraints.emplace_back(Polynomial("i_0") + 2 * Polynomial("k_0"), 5);
  constraints.emplace_back(Polynomial("i_0") + Polynomial("i_1") + Polynomial("k_0"), 35);
  constraints.emplace_back(Polynomial("i_0") + 2 * Polynomial("i_1"), 70);
  ILPSolver solver;
  EXPECT_TRUE(solver.solve(constraints, Polynomial("i_0")));
  EXPECT_EQ(solver.reportObjective(), 0);
  EXPECT_TRUE(solver.solve(constraints, -Polynomial("i_0")));
  EXPECT_EQ(solver.reportObjective(), -1);
  EXPECT_TRUE(solver.solve(constraints, Polynomial("i_1")));
  EXPECT_EQ(solver.reportObjective(), 0);
  EXPECT_TRUE(solver.solve(constraints, -Polynomial("i_1")));
  EXPECT_EQ(solver.reportObjective(), -34);
  EXPECT_TRUE(solver.solve(constraints, Polynomial("k_0")));
  EXPECT_EQ(solver.reportObjective(), 0);
  EXPECT_TRUE(solver.solve(constraints, -Polynomial("k_0")));
  EXPECT_EQ(solver.reportObjective(), -2);
}
}  // namespace milp
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
