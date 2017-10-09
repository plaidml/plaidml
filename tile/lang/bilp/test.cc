#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tile/lang/bignum.h"
#include "tile/lang/bilp/ilp_solver.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace bilp {

TEST(BilpTest, TestTest) { EXPECT_EQ(0, 0); }

TEST(BilpTest, RationalTest) { EXPECT_EQ(Rational(0, 1), 0); }

TEST(BilpTest, BasicTableauTest) {
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

TEST(BilpTest, OptimizeCanonicalTest) {
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
  EXPECT_EQ(t.makeOptimal(true), true);

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

TEST(BilpTest, SimpleOptimizeTest) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(Polynomial("x") + 4, 4);
  Polynomial obj = 3 * Polynomial("x");
  ILPSolver solver;
  Tableau t = solver.makeStandardFormTableau(constraints, obj);
  EXPECT_EQ(t.makeOptimal(), true);

  std::vector<Rational> soln = t.getSymbolicSolution();

  EXPECT_EQ(t.varNames()[0], "_x_pos");
  EXPECT_EQ(t.varNames()[1], "_x_neg");
  EXPECT_EQ(soln[0], 0);
  EXPECT_EQ(soln[1], 4);
}

TEST(BilpTest, OptimizeTest2D) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(Polynomial("x") + Polynomial("y") + 2, 4);
  constraints.emplace_back(Polynomial("x") + 1, 4);
  constraints.emplace_back(Polynomial("y") + 2, 5);
  Polynomial obj = -3 * Polynomial("x") + 2 * Polynomial("y");
  ILPSolver solver;
  Tableau t = solver.makeStandardFormTableau(constraints, obj);
  EXPECT_EQ(t.makeOptimal(), true);

  std::vector<Rational> soln = t.getSymbolicSolution();

  EXPECT_EQ(t.varNames()[0], "_x_pos");
  EXPECT_EQ(t.varNames()[1], "_x_neg");
  EXPECT_EQ(t.varNames()[2], "_y_pos");
  EXPECT_EQ(t.varNames()[3], "_y_neg");
  EXPECT_EQ(soln[0], 2);
  EXPECT_EQ(soln[1], 0);
  EXPECT_EQ(soln[2], 0);
  EXPECT_EQ(soln[3], 2);
}

TEST(BilpTest, TrivialILPTest) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(Polynomial("x") + Polynomial("y") + 2, 4);
  constraints.emplace_back(Polynomial("x") + 1, 4);
  constraints.emplace_back(Polynomial("y") + 2, 5);
  Polynomial obj = -3 * Polynomial("x") + 2 * Polynomial("y");
  ILPSolver solver;
  Tableau t = solver.makeStandardFormTableau(constraints, obj);
  ILPResult res = solver.solve(t);

  EXPECT_EQ(res.soln["x"], 2);
  EXPECT_EQ(res.soln["y"], -2);

  EXPECT_EQ(res.obj_val, -10);
}

TEST(BilpTest, ILPTest2D) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(2 * Polynomial("x") + Polynomial("y") + 2, 6);
  constraints.emplace_back(Polynomial("x") + 1, 4);
  constraints.emplace_back(Polynomial("y") + 2, 5);
  Polynomial obj = -3 * Polynomial("x") + 2 * Polynomial("y");
  ILPSolver solver;
  ILPResult res = solver.solve(constraints, obj);

  EXPECT_EQ(res.soln["x"], 2);
  EXPECT_EQ(res.soln["y"], -2);

  EXPECT_EQ(res.obj_val, -10);
}

TEST(BilpTest, Subdivision1D) {
  std::vector<RangeConstraint> constraints;
  constraints.emplace_back(Polynomial("i_0"), 2);
  constraints.emplace_back(Polynomial("i_0") + 2 * Polynomial("k_0"), 5);
  constraints.emplace_back(Polynomial("i_0") + Polynomial("i_1") + Polynomial("k_0"), 35);
  constraints.emplace_back(Polynomial("i_0") + 2 * Polynomial("i_1"), 70);

  std::vector<Polynomial> objectives;
  objectives.emplace_back(Polynomial("i_0"));
  objectives.emplace_back(-Polynomial("i_0"));
  objectives.emplace_back(Polynomial("i_1"));
  objectives.emplace_back(-Polynomial("i_1"));
  objectives.emplace_back(Polynomial("k_0"));
  objectives.emplace_back(-Polynomial("k_0"));
  ILPSolver solver;
  std::map<Polynomial, ILPResult> res = solver.batch_solve(constraints, objectives);

  EXPECT_EQ(res[Polynomial("i_0")].obj_val, 0);
  EXPECT_EQ(res[-Polynomial("i_0")].obj_val, -1);
  EXPECT_EQ(res[Polynomial("i_1")].obj_val, 0);
  EXPECT_EQ(res[-Polynomial("i_1")].obj_val, -34);
  EXPECT_EQ(res[Polynomial("k_0")].obj_val, 0);
  EXPECT_EQ(res[-Polynomial("k_0")].obj_val, -2);
}

TEST(MilpTest, RandomConstraintsTest) {
  const int varSize = 8;
  for (size_t test_count = 0; test_count < 20; ++test_count) {
    std::vector<RangeConstraint> constraints;
    constraints.emplace_back(Polynomial("x"), varSize);
    constraints.emplace_back(Polynomial("y"), varSize);
    constraints.emplace_back(Polynomial("z"), varSize);
    constraints.emplace_back(Polynomial("w"), varSize);
    for (int i = 0; i < rand() % 4 + 2; ++i) {                           // NOLINT (runtime/threadsafe_fn)
      Rational x_coeff = Rational(rand() % 9 - 4, rand() % 4 + 1);       // NOLINT (runtime/threadsafe_fn)
      Rational y_coeff = Rational(rand() % 11 - 5, rand() % 3 + 1);      // NOLINT (runtime/threadsafe_fn)
      Rational z_coeff = Rational(rand() % 11 - 5, rand() % 5 + 1);      // NOLINT (runtime/threadsafe_fn)
      Rational w_coeff = Rational(rand() % 15 - 7, rand() % 4 + 1);      // NOLINT (runtime/threadsafe_fn)
      Rational const_term = Rational(rand() % 31 - 20, rand() % 4 + 1);  // NOLINT (runtime/threadsafe_fn)
      constraints.emplace_back(x_coeff * Polynomial("x") + y_coeff * Polynomial("y") + z_coeff * Polynomial("z") +
                                   w_coeff * Polynomial("w") + const_term,
                               rand() % 10 + 12);
    }

    Integer x_min = varSize;
    Integer x_max = -1;
    Integer y_min = varSize;
    Integer y_max = -1;
    Integer z_min = varSize;
    Integer z_max = -1;
    Integer w_min = varSize;
    Integer w_max = -1;
    // Solve via brute force
    bool is_feasible = false;
    std::map<std::string, Rational> values;
    for (int x = 0; x < varSize; ++x) {
      values["x"] = x;
      for (int y = 0; y < varSize; ++y) {
        values["y"] = y;
        for (int z = 0; z < varSize; ++z) {
          values["z"] = z;
          for (int w = 0; w < varSize; ++w) {
            values["w"] = w;
            bool is_feasible_here = true;
            for (const RangeConstraint& c : constraints) {
              Rational value = c.poly.eval(values);
              if (value < 0 || c.range <= value || value != Floor(value)) {
                is_feasible_here = false;
                break;
              }
            }
            if (is_feasible_here) {
              is_feasible = true;
              if (x < x_min) {
                x_min = x;
              }
              if (x > x_max) {
                x_max = x;
              }
              if (y < y_min) {
                y_min = y;
              }
              if (y > y_max) {
                y_max = y;
              }
              if (z < z_min) {
                z_min = z;
              }
              if (z > z_max) {
                z_max = z;
              }
              if (w < w_min) {
                w_min = w;
              }
              if (w > w_max) {
                w_max = w;
              }
            }
          }
        }
      }
    }
    std::vector<Polynomial> objectives;
    objectives.emplace_back(Polynomial("x"));
    objectives.emplace_back(Polynomial("x", -1));
    objectives.emplace_back(Polynomial("y"));
    objectives.emplace_back(Polynomial("y", -1));
    objectives.emplace_back(Polynomial("z"));
    objectives.emplace_back(Polynomial("z", -1));
    objectives.emplace_back(Polynomial("w"));
    objectives.emplace_back(Polynomial("w", -1));
    if (is_feasible) {
      ILPSolver solver;
      std::map<Polynomial, ILPResult> result = solver.batch_solve(constraints, objectives);
      for (const auto& kvp : result) {
        std::string var = kvp.first.GetNonzeroIndex();
        Rational obj_val = kvp.second.obj_val;
        if (obj_val < 0 || obj_val >= varSize) {
          // This result would not have been found in brute force search, so skip
          continue;
        }
        if (kvp.first[var] == 1) {
          if (var == "x") {
            EXPECT_EQ(x_min, obj_val);
          } else if (var == "y") {
            EXPECT_EQ(y_min, obj_val);
          } else if (var == "z") {
            EXPECT_EQ(z_min, obj_val);
          } else if (var == "w") {
            EXPECT_EQ(w_min, obj_val);
          } else {
            throw std::logic_error("Unexpected variable in RandomConstraintsTest");
          }
        } else if (kvp.first[var] == -1) {
          if (var == "x") {
            EXPECT_EQ(x_max, -obj_val);
          } else if (var == "y") {
            EXPECT_EQ(y_max, -obj_val);
          } else if (var == "z") {
            EXPECT_EQ(z_max, -obj_val);
          } else if (var == "w") {
            EXPECT_EQ(w_max, -obj_val);
          } else {
            throw std::logic_error("Unexpected variable in RandomConstraintsTest");
          }
        }
      }
    }
  }
}
}  // namespace bilp
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
