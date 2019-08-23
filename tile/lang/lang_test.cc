#include <mutex>

#include "tile/base/shape.h"
#include "tile/lang/bound.h"
#include "tile/lang/compile.h"
#include "tile/lang/defract.h"
#include "tile/lang/flat.h"
#include "tile/lang/gen_contract.h"
#include "tile/lang/generate.h"
#include "tile/lang/loop.h"
#include "tile/lang/ops.h"
#include "tile/lang/parser.h"
#include "tile/lang/reduce.h"
#include "tile/lang/replace.h"
#include "tile/lang/sembuilder.h"
#include "tile/lang/semprinter.h"
#include "tile/lang/semtree.h"
#include "tile/lang/sym_poly.h"
#include "tile/lang/symbolic.h"
#include "tile/lang/tile_opt.h"
#include "tile/lang/type.h"
#include "tile/math/matrix.h"

#include "base/util/catch.h"
#include "base/util/logging.h"

namespace vertexai {
namespace tile {
namespace lang {

using namespace math;  // NOLINT

namespace {

const HardwareSettings& TestGPU() {
  static HardwareSettings settings;
  static std::once_flag init;

  std::call_once(init, [&] {
    settings.threads = 256;
    settings.vec_size = 1;
    settings.use_global = false;
    settings.mem_width = 32;
    settings.max_local_mem = 18 * 1024;
    settings.max_global_mem = 2145386496;
    settings.max_regs = 18 * 1024;
    settings.goal_groups = 20;
    settings.goal_flops_per_byte = 20;
    settings.goal_dimension_sizes.push_back(1024);
    settings.goal_dimension_sizes.push_back(1024);
    settings.goal_dimension_sizes.push_back(1024);
  });

  return settings;
}

TEST_CASE("Bound check convolution edged", "[bound]") {
  Polynomial<Rational> x("x"), i("i");
  std::vector<RangeConstraint> cons = {{x, 5}, {x + i, 7}, {i, 3}};
  IndexBounds out;
  std::vector<SimpleConstraint> rcons;
  std::tie(out, rcons) = ComputeBounds(cons);
  REQUIRE(rcons.size() == 0);
  REQUIRE(out.size() == 2);
  REQUIRE(out["i"].min == 0);
  REQUIRE(out["i"].max == 2);
  REQUIRE(out["x"].min == 0);
  REQUIRE(out["x"].max == 4);
}

TEST_CASE("Bound check convolution equal sized", "[bound]") {
  Polynomial<Rational> x("x"), i("i");
  std::vector<RangeConstraint> cons = {{x, 7}, {x + i, 7}, {i + 1, 3}};
  IndexBounds out;
  std::vector<SimpleConstraint> rcons;
  std::tie(out, rcons) = ComputeBounds(cons);
  REQUIRE(rcons.size() == 2);
  REQUIRE(rcons[0].poly == (-x + -i));
  REQUIRE(rcons[0].rhs == 0);
  REQUIRE(rcons[1].poly == (x + i));
  REQUIRE(rcons[1].rhs == 6);
  REQUIRE(out.size() == 2);
  REQUIRE(out["i"].min == -1);
  REQUIRE(out["i"].max == 1);
  REQUIRE(out["x"].min == 0);
  REQUIRE(out["x"].max == 6);
}

TEST_CASE("Optimization of Matrix Multiply", "[mat_opt][opt]") {
  Parser p;
  auto c = p.ParseContraction("O[i,j] = +(A[i,k] * B[k,j])");
  FlatContraction f =
      Flatten(c, {SimpleShape(DataType::FLOAT32, {1024, 1024}), SimpleShape(DataType::FLOAT32, {1024, 1024}),
                  SimpleShape(DataType::FLOAT32, {1024, 1024})});
  uint64_t one = 1;
  std::vector<uint64_t> best_tile = {1, 1, 1};
  double best_score = 0;

  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 10; j++) {
      for (size_t k = 0; k < 10; k++) {
        std::vector<uint64_t> tile = {one << i, one << j, one << k};
        double score = ComputeScore(TestGPU(), ComputeTileStats(TestGPU(), f, tile));
        if (score > best_score) {
          best_tile = tile;
          best_score = score;
        }
      }
    }
  }
  IVLOG(1, "Best Score = " << best_score << " " << best_tile);
  std::multimap<double, std::vector<uint64_t>> out = TileOptimize(TestGPU(), f, false);
  auto it = out.rbegin();
  IVLOG(1, "Opt Score = " << it->first << " " << it->second);
  REQUIRE(it->first == best_score);
}

TEST_CASE("Subdivision 1D input width 2**n", "[subdivision]") {
  const std::size_t kernelSize = 5;

  for (std::size_t inputSize = 35; inputSize < 100; inputSize += 2) {
    // for (unsigned long inputSize = 512; inputSize < 65536; inputSize *= 2) {
    Parser p;
    Contraction c = p.ParseContraction("O[i] = +(I[i/2 + k/2] * K[k])");

    FlatContraction f =
        Compile(c, {SimpleShape(DataType::FLOAT32, {2 * inputSize}), SimpleShape(DataType::FLOAT32, {inputSize}),
                    SimpleShape(DataType::FLOAT32, {kernelSize})});
    IVLOG(1, "Flat:\n" << f.toString());
    REQUIRE(f.ranges[0] == 2);
    REQUIRE(f.ranges[1] == inputSize);
    REQUIRE(f.ranges[2] == (kernelSize + 1) / 2);
  }
}

TEST_CASE("Awkward Flatten") {
  Parser p;
  Contraction c = p.ParseContraction("X_T10[24*n0 + 8*n1 + 4*n2 + n3 : 168] = +(X_I_0[n0, n1, n2, n3])");
  FlatContraction f = Compile(c, {
                                     SimpleShape(DataType::FLOAT32, {168}),
                                     SimpleShape(DataType::FLOAT32, {7, 3, 2, 4}),
                                 });
  IVLOG(1, "Flat:\n" << f.toString());
}

TEST_CASE("Optimization of Convolution", "[conv_opt][opt]") {
  Parser p;
  auto c = p.ParseContraction("O[n, x, y, co] = +(K[i, j, co, ci] * I[n, x+i, y+j, ci])");
  FlatContraction op = Flatten(c, {
                                      SimpleShape(DataType::FLOAT32, {128, 25, 25, 384}),
                                      SimpleShape(DataType::FLOAT32, {3, 3, 384, 256}),
                                      SimpleShape(DataType::FLOAT32, {128, 27, 27, 256}),
                                  });
  IVLOG(1, "Flat:\n" << op.toString());
  auto settings = TestGPU();
  auto vectorized_op = Vectorize(op, settings.vec_size);
  auto by_score = TileOptimize(settings, vectorized_op, true);
  auto tile = by_score.rbegin()->second;
  double score2 = ComputeScore(settings, ComputeTileStats(TestGPU(), op, tile));
  IVLOG(1, "Opt Score = " << score2 << " " << tile);
  REQUIRE(score2 > .4);
}

TEST_CASE("Vectorized Flop Computation", "[conv_opt][opt]") {
  Parser p;
  auto c = p.ParseContraction("O[n, x, y, co] = +(K[i, j, co, ci] * I[n, x+i, y+j, ci])");
  FlatContraction op = Flatten(c, {
                                      SimpleShape(DataType::FLOAT32, {128, 25, 25, 384}),
                                      SimpleShape(DataType::FLOAT32, {3, 3, 384, 256}),
                                      SimpleShape(DataType::FLOAT32, {128, 27, 27, 256}),
                                  });
  uint64_t ops = 128ll * 3 * 3 * 25 * 25 * 384 * 256 * 2;
  auto settings = TestGPU();
  auto vectorized_gpu = TestGPU();
  vectorized_gpu.vec_size = 4;
  auto vectorized_op = Vectorize(op, settings.vec_size);
  auto by_score = TileOptimize(settings, vectorized_op, true);
  auto tile = by_score.rbegin()->second;
  proto::PerfStats psn = ComputeTileStats(settings, op, tile);
  proto::PerfStats psv = ComputeTileStats(vectorized_gpu, op, tile);
  REQUIRE(psn.true_ops() == psv.true_ops());
  REQUIRE(psn.true_ops() == ops);
}

TEST_CASE("Compile Agg Prod", "[compile]") {
  Parser p;
  Contraction c = p.ParseContraction("O[] = *(A[x,y])");
  std::vector<TensorShape> shapes = {{DataType::FLOAT32, {}}, {DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}}};
  FlatContraction r = Compile(c, shapes);
  REQUIRE(r.names == (std::vector<std::string>{"x", "y"}));
  REQUIRE(r.constraints.size() == 0);
  REQUIRE(r.agg_type == DataType::FLOAT32);
  REQUIRE(r.agg_op == AggregationOp::PROD);
  REQUIRE(r.comb_op == CombinationOp::MULTIPLY);
  REQUIRE(r.ranges == (std::vector<uint64_t>{100, 100}));
  REQUIRE(r.access[0].offset == 0);
  REQUIRE(r.access[1].offset == 0);
  REQUIRE(r.access[0].strides == (std::vector<int64_t>{0, 0}));
  REQUIRE(r.access[1].strides == (std::vector<int64_t>{128, 1}));
}

TEST_CASE("Compile MatMul", "[compile]") {
  Parser p;
  Contraction c = p.ParseContraction("O[i,j] = +(A[i,k] * B[k,j])");
  std::vector<TensorShape> shapes = {{DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}},
                                     {DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}},
                                     {DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}}};
  FlatContraction r = Compile(c, shapes);
  REQUIRE(r.names == (std::vector<std::string>{"i", "j", "k"}));
  REQUIRE(r.constraints.size() == 0);
  REQUIRE(r.agg_type == DataType::FLOAT32);
  REQUIRE(r.agg_op == AggregationOp::SUM);
  REQUIRE(r.comb_op == CombinationOp::MULTIPLY);
  REQUIRE(r.ranges == (std::vector<uint64_t>{100, 100, 100}));
  REQUIRE(r.access[0].offset == 0);
  REQUIRE(r.access[1].offset == 0);
  REQUIRE(r.access[2].offset == 0);
  REQUIRE(r.access[0].strides == (std::vector<int64_t>{128, 1, 0}));
  REQUIRE(r.access[1].strides == (std::vector<int64_t>{128, 0, 1}));
  REQUIRE(r.access[2].strides == (std::vector<int64_t>{0, 1, 128}));
}

TEST_CASE("Compile unpool overlap", "[compile][unpool]") {
  Parser p;
  Contraction c = p.ParseContraction("O[2*x + i] = +(I[x]), i < 4");
  std::vector<TensorShape> shapes = {
      {DataType::FLOAT32, {{1, 100UL}}},
      {DataType::FLOAT32, {{1, 50UL}}},
  };
  FlatContraction r = Compile(c, shapes);
  REQUIRE(r.names == (std::vector<std::string>{"v0_0", "v0_1", "v1_0"}));
  REQUIRE(r.constraints.size() == 1);
  REQUIRE(r.constraints[0].lhs == (std::vector<int64_t>{0, -1, 1}));
  REQUIRE(r.constraints[0].rhs == 0);
  REQUIRE(r.agg_type == DataType::FLOAT32);
  REQUIRE(r.agg_op == AggregationOp::SUM);
  REQUIRE(r.comb_op == CombinationOp::MULTIPLY);
  REQUIRE(r.ranges == (std::vector<uint64_t>{2, 50, 2}));
  REQUIRE(r.access[0].offset == 0);
  REQUIRE(r.access[1].offset == 0);
  REQUIRE(r.access[0].strides == (std::vector<int64_t>{1, 2, 0}));
  REQUIRE(r.access[1].strides == (std::vector<int64_t>{0, 1, -1}));
}

TEST_CASE("Compile unpool clean", "[compile][unpool]") {
  Parser p;
  Contraction c = p.ParseContraction("O[2*x + i] = +(I[x]), i < 2");
  std::vector<TensorShape> shapes = {
      {DataType::FLOAT32, {{1, 100UL}}},
      {DataType::FLOAT32, {{1, 50UL}}},
  };
  FlatContraction r = Compile(c, shapes);
  REQUIRE(r.names == (std::vector<std::string>{"v0_0", "v0_1"}));
  REQUIRE(r.constraints.size() == 0);
  REQUIRE(r.agg_type == DataType::FLOAT32);
  REQUIRE(r.agg_op == AggregationOp::SUM);
  REQUIRE(r.comb_op == CombinationOp::MULTIPLY);
  REQUIRE(r.ranges == (std::vector<uint64_t>{2, 50}));
  REQUIRE(r.access[0].offset == 0);
  REQUIRE(r.access[1].offset == 0);
  REQUIRE(r.access[0].strides == (std::vector<int64_t>{1, 2}));
  REQUIRE(r.access[1].strides == (std::vector<int64_t>{0, 1}));
}

TEST_CASE("Compile strided convolution derivate", "[compile][conv_d]") {
  Parser p;
  Contraction c = p.ParseContraction("DA[n, i + x, j + 2*y, ci] = +(DC[n, x, y, co] * B[i, j, ci, co])");
  std::vector<TensorShape> shapes = {
      SimpleShape(DataType::FLOAT32, {1, 3, 6, 1}),
      SimpleShape(DataType::FLOAT32, {1, 2, 3, 1}),
      SimpleShape(DataType::FLOAT32, {2, 2, 1, 1}),
  };
  FlatContraction r = Compile(c, shapes);
  // This is testing against regression, none of these were worked out from first principles
  REQUIRE(r.names == (std::vector<std::string>{"v1_0", "v2_0", "v2_1", "v5_0"}));
  REQUIRE(r.constraints.size() == 2);
  REQUIRE(r.constraints[0].lhs == (std::vector<int64_t>{-1, 0, 0, 1}));
  REQUIRE(r.constraints[0].rhs == 0);
  REQUIRE(r.constraints[1].lhs == (std::vector<int64_t>{1, 0, 0, -1}));
  REQUIRE(r.constraints[1].rhs == 1);
  REQUIRE(r.ranges == (std::vector<uint64_t>{3, 2, 3, 2}));
  REQUIRE(r.access[0].strides == (std::vector<int64_t>{6, 1, 2, 0}));
  REQUIRE(r.access[1].strides == (std::vector<int64_t>{0, 0, 1, 3}));
  REQUIRE(r.access[2].strides == (std::vector<int64_t>{2, 1, 0, -2}));
}

TEST_CASE("Flatten wacky matrix multiply type thing", "[flatten]") {
  Contraction c(2);
  c.comb_op = CombinationOp::MULTIPLY;
  c.agg_op = AggregationOp::SUM;
  Polynomial<Rational> i("i"), j("j"), k("k");
  c.specs[0].spec = {i, j};
  c.specs[1].spec = {i + 2, k};
  c.specs[2].spec = {k, j};
  c.constraints.push_back(SymbolicConstraint({k, 50}));
  c.constraints.push_back(SymbolicConstraint({i + j, 75}));
  FlatContraction fc = Flatten(c, {{{DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}},
                                    {DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}},
                                    {DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}}}});

  for (size_t i = 0; i < 3; i++) {
    REQUIRE(fc.access[i].type == DataType::FLOAT32);
  }
  REQUIRE(fc.agg_type == DataType::FLOAT32);
  REQUIRE(fc.comb_op == CombinationOp::MULTIPLY);
  REQUIRE(fc.agg_op == AggregationOp::SUM);
  REQUIRE(fc.constraints.size() == 1);
  REQUIRE(fc.constraints[0].lhs == (std::vector<int64_t>{1, 1, 0}));
  REQUIRE(fc.constraints[0].rhs == 74);
  REQUIRE(fc.names == (std::vector<std::string>{"i", "j", "k"}));
  REQUIRE(fc.ranges == (std::vector<uint64_t>{75, 75, 50}));
  REQUIRE(fc.access[0].strides == (std::vector<int64_t>{128, 1, 0}));
  REQUIRE(fc.access[1].strides == (std::vector<int64_t>{128, 0, 1}));
  REQUIRE(fc.access[2].strides == (std::vector<int64_t>{0, 1, 128}));
  REQUIRE(fc.access[0].offset == 0);
  REQUIRE(fc.access[1].offset == 256);
  REQUIRE(fc.access[2].offset == 0);
}

TEST_CASE("Whole ball of wax", "[emit]") {
  // NOTE: This test doesn't test anything.  It's really for staring at generated code
  // when changing things...
  Parser p;
  Program prog = p.Parse("function (A[I,K], B[K,J]) -> (O) { O[i,j : I,J] = +(A[i,k] * B[k,j]); }");
  ShapeMap inputs;
  ShapeMap outputs;
  outputs.emplace("O", TensorShape{DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}});
  inputs.emplace("A", TensorShape{DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}});
  inputs.emplace("B", TensorShape{DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}});
  TileOptimizer optimizer;
  KernelList r = GenerateProgram(prog, inputs, outputs, TestGPU(), optimizer);
}

TEST_CASE("Two outputs", "[multiout]") {
  Parser p;
  Program prog = p.Parse("function (I[N]) -> (O1, O2) { O1 = I; O2 = I; }");
  ShapeMap inputs;
  ShapeMap outputs;
  outputs.emplace("O1", SimpleShape(DataType::FLOAT32, {3}));
  outputs.emplace("O2", SimpleShape(DataType::FLOAT32, {3}));
  inputs.emplace("I", SimpleShape(DataType::FLOAT32, {3}));
  TileOptimizer optimizer;
  KernelList r = GenerateProgram(prog, inputs, outputs, TestGPU(), optimizer);
}

TEST_CASE("Gausian Elimination 1", "[reduce]") {
  Polynomial<Rational> i("i"), j("j"), k("k"), x("x"), y("y");
  Contraction op(2);
  op.specs[0].spec = {k + j, 2 * x + i + 3, j - x, 2 * k + 2 * j};
  op.specs[1].spec = {5 * i, -3 * j + x, y};
  op.specs[2].spec = {x, j - k, i * 2, y + j};
  op.constraints = {SymbolicConstraint({k, 5}), SymbolicConstraint({j + k, 1})};

  Contraction new_op = ReduceOutputPolynomials(op, {{j + k, 1}, {k, 5}, {5 * i, 100}, {y, 100}});
}

TEST_CASE("Doc examples", "[doc]") {
  Polynomial<Rational> i("i"), j("j"), k("k");
  Contraction op(2);
  op.specs[0].spec = {k, 2 * k + 5, k - 2 * j};
  op.specs[1].spec = {5 * i - 2, -3 * j};
  op.specs[2].spec = {2 * i + k, 3 * k};
  op.constraints = {SymbolicConstraint({i, 5})};

  Contraction reduced = ReduceOutputPolynomials(op, {{2 * i + k, 1}});
  IVLOG(2, "Reduced:\n" << to_string(reduced));
  std::vector<TensorShape> shapes = {SimpleShape(DataType::FLOAT32, {7, 10, 6}),
                                     SimpleShape(DataType::FLOAT32, {10, 10}),
                                     SimpleShape(DataType::FLOAT32, {10, 10})};
  std::vector<RangeConstraint> cons = GatherConstraints(reduced, shapes);
  Contraction defract = Defract(reduced, cons);
  IVLOG(2, "Defracted:\n" << to_string(defract));
  FlatContraction flat = Flatten(defract, shapes);
}

TEST_CASE("MatMulBadGrad", "[reduce]") {
  Polynomial<Rational> i("i"), y("y"), x("x");
  Contraction op(2);

  op.specs[0].spec = {y, i};
  op.specs[1].spec = {y, x};
  op.specs[2].spec = {x, i};

  Contraction new_op = ReduceOutputPolynomials(op, {{x, 5}});
}

TEST_CASE("Functions", "[compile]") {
  Op op;
  Program p;
  p.inputs.push_back(Input{Input::VARIABLE, "x"});
  p.outputs.emplace_back("y");
  op.tag = Op::FUNCTION;
  op.f.fn = "exp";
  op.output = "y";
  op.inputs.push_back("x");
  p.ops.push_back(op);
  std::map<std::string, TensorShape> inputs;
  std::map<std::string, TensorShape> outputs;
  inputs.emplace("x", TensorShape{DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}});
  outputs.emplace("y", TensorShape{DataType::FLOAT32, {{128L, 100UL}, {1L, 100UL}}});
  TileOptimizer optimizer;
  KernelList result = GenerateProgram(p, inputs, outputs, TestGPU(), optimizer);
  sem::Print emit(*result.kernels[0].kfunc);
  std::string code = emit.str();
  REQUIRE(code.find("exp") != std::string::npos);
}

TEST_CASE("Program Parsing", "[parsing]") {
  Parser parser;
  Program prog = parser.Parse(R"***(
      function (A[X], B[X]) -> (THREE) {
        ONE[] = +(A[x]);
        TWO[] = +(A[x] * B[y]), x+ y< 9, x < 2;
        THREE = relu(TWO);
      }
  )***");

  REQUIRE(prog.ops[0].tag == Op::CONTRACTION);
  REQUIRE(prog.ops[0].c.specs[0].id == "ONE");
  REQUIRE(prog.ops[0].c.specs.size() == 2);

  REQUIRE(prog.ops[1].tag == Op::CONSTANT);
  REQUIRE(prog.ops[2].tag == Op::CONSTANT);

  REQUIRE(prog.ops[3].tag == Op::CONTRACTION);
  REQUIRE(prog.ops[3].c.specs[0].id == "TWO");
  REQUIRE(prog.ops[3].c.specs.size() == 3);

  REQUIRE(prog.ops[4].tag == Op::FUNCTION);
  REQUIRE(prog.ops[4].inputs[0] == "TWO");
  REQUIRE(prog.ops[4].output == "THREE");
  REQUIRE(prog.ops[4].f.fn == "relu");
}

TEST_CASE("Function", "[parsing]") {
  Parser parser;
  Program prog = parser.Parse("function (A) -> (B) { B = relu(A); }");
  REQUIRE(prog.ops[0].tag == Op::FUNCTION);
  REQUIRE(prog.ops[0].f.fn == "relu");
  REQUIRE(prog.ops[0].inputs[0] == "A");
  REQUIRE(prog.ops[0].output == "B");
}

TEST_CASE("Naked Polynomial<Rational>", "[parsing]") {
  Parser parser;
  REQUIRE(parser.ParsePolynomial("3*x-i+4").toString() == "4 - i + 3*x");
  REQUIRE(parser.ParsePolynomial("x*3-i+4").toString() == "4 - i + 3*x");
  REQUIRE(parser.ParsePolynomial("x/3-i+4").toString() == "4 - i + 1/3*x");
  REQUIRE(parser.ParsePolynomial("(x-i+4)/3").toString() == "4/3 - 1/3*i + 1/3*x");
  REQUIRE(parser.ParsePolynomial("(x/3-i)/5+4").toString() == "4 - 1/5*i + 1/15*x");
  REQUIRE(parser.ParsePolynomial("(x-1)/3").toString() == "-1/3 + 1/3*x");
}

TEST_CASE("Type checking", "[type]") {
  Parser parser;
  Program prog = parser.Parse(R"***(
      function (Input[D, I], Weights[O, I], Biases[O], ReluLeak) -> (Output) {
        O1[d, o : D, O] = +(Weights[o, i] * Input[d, i]);
        O2[d, o : D, O] = +(O1[d, o] * Biases[o]);
        Output = (O2 < 0 ? ReluLeak * O2 : O2);
      }
  )***");
  IVLOG(1, prog.ops);
  Bindings types;
  types.emplace("Input", Binding(SimpleShape(DataType::FLOAT32, {64, 77})));
  types.emplace("Weights", Binding(SimpleShape(DataType::FLOAT32, {55, 77})));
  types.emplace("Biases", Binding(SimpleShape(DataType::INT32, {55})));
  types.emplace("ReluLeak", Binding(.05, DataType::FLOAT32));
  TypeCheck(&prog, &types);
  IVLOG(1, "Types " << types);
  REQUIRE(types.at("Output").shape == SimpleShape(DataType::FLOAT32, {64, 55}));
  REQUIRE(types.at("D").iconst == 64);
  REQUIRE(types.at("I").iconst == 77);
  REQUIRE(types.at("O").iconst == 55);
}

TEST_CASE("Mean", "[type]") {
  Parser parser;
  Program prog = parser.Parse(R"***(
      function (Input[D]) -> (Mean) {
        Sum[] = +(Input[i]);
        Mean = Sum / D;
      }
  )***");
  IVLOG(1, prog.ops);
  Bindings types;
  types.emplace("Input", Binding(SimpleShape(DataType::FLOAT32, {101})));
  TypeCheck(&prog, &types);
  IVLOG(1, "Types " << types);
  IVLOG(1, prog.ops);
  REQUIRE(types.at("Mean").shape == SimpleShape(DataType::FLOAT32, {}));
}

TEST_CASE("JustRelu", "[emit]") {
  Parser parser;
  Program prog = parser.Parse("function (X[N]) -> (Y) { Y = (X < 0 ? 0 : X); }");
  IVLOG(1, prog.ops);
  ShapeMap inputs;
  ShapeMap outputs;
  inputs.emplace("X", SimpleShape(DataType::FLOAT32, {100}));
  outputs.emplace("Y", SimpleShape(DataType::FLOAT32, {100}));
  TileOptimizer optimizer;
  KernelList r = GenerateProgram(prog, inputs, outputs, TestGPU(), optimizer, "ID");
  REQUIRE(r.kernels.size() == 1);
  REQUIRE(r.kernels[0].inputs == std::vector<std::string>({"X"}));
  REQUIRE(r.kernels[0].outputs == std::vector<std::string>({"Y"}));
}

TEST_CASE("NoRedeclare", "[emit]") {
  Parser parser;
  Program prog = parser.Parse("function (X[N]) -> (X) { X = 2*X; }");
  ShapeMap stuff;
  stuff.emplace("x", SimpleShape(DataType::FLOAT32, {100}));
  TileOptimizer optimizer;
  REQUIRE_THROWS(GenerateProgram(prog, stuff, stuff, TestGPU(), optimizer, "ID"));
}

TEST_CASE("Ast", "[ast]") {
  using namespace sem::builder;  // NOLINT
  sem::Type idxType = {sem::Type::INDEX};
  auto n = _("n");
  auto i = _("i");
  auto r = _("r");
  auto f = _Function("factorial", idxType, {{idxType, "n"}},
                     {_DeclareConst(idxType, "r", 1),
                      _Block({
                          _DeclareConst(idxType, "i", 1),
                          _While(i <= n, _Block({r = r * i, i = i + 1})),
                      }),
                      _Return(r)});

  sem::Print emit(*f);
  IVLOG(1, "Code:\n" << emit.str());
}

TEST_CASE("Softmax Deriv", "[deriv]") {
  Parser p;
  auto softmax = std::make_shared<BoundFunction>(R"***(
      function (IN[X,Y]) -> (OUT) {
        OUT = builtin_softmax(IN, X, Y);
      }
  )***");

  auto crossentropy = std::make_shared<BoundFunction>(R"***(
      function (Y[I,J], TY[I,J]) -> (E) {
        P = -log(Y) * TY;
	E[] = +(P[i, j]);
      }
  )***");

  FunctionApplication sm(softmax);
  FunctionApplication ce(crossentropy);

  auto x = std::make_shared<PlaceholderValue>(2);
  auto y = std::make_shared<PlaceholderValue>(2);
  sm.SetInput("IN", x);
  ce.SetInput("Y", sm.GetOutput("OUT"));
  ce.SetInput("TY", y);
  auto e = ce.GetOutput("E");

  Gradient grad(e);
  auto dx = grad(x);

  auto ofunc = std::make_shared<BoundFunction>();
  ofunc->AddInput("X", x);
  ofunc->AddInput("Y", y);
  ofunc->AddOutput("DX", dx);
  ofunc->Done();

  IVLOG(1, to_string(ofunc->prog()));

  BoundFunction rfunc;
  FunctionApplication fo(ofunc);
  TensorShape ss = SimpleShape(DataType::FLOAT32, {50, 100});
  fo.SetInput("X", TensorValue::make(std::make_shared<BufferBase>(), ss));
  fo.SetInput("Y", TensorValue::make(std::make_shared<BufferBase>(), ss));
  rfunc.AddUpdate(TensorValue::make(std::make_shared<BufferBase>(), ss), fo.GetOutput("DX"));
  rfunc.Done();
  auto ri = rfunc.PrepareToRun("test");

  Parser parse;
  Program prog = parse.Parse(ri.code);
  TileOptimizer optimizer;
  auto cr = GenerateProgram(prog, ri.input_shapes, ri.output_shapes, TestGPU(), optimizer, "test");
}

TEST_CASE("Function Deriv", "[deriv]") {
  auto f = std::make_shared<BoundFunction>(R"***(
    function (X) -> (Y) {
      Y = 3*X*X + exp(X);
    }
  )***");
  auto x = std::make_shared<PlaceholderValue>(0);
  FunctionApplication app(f);
  app.SetInput("X", x);
  auto y = app.GetOutput("Y");

  Gradient grad(y);
  auto dx = grad(x);

  BoundFunction ofunc;
  ofunc.AddInput("X", x);
  ofunc.AddOutput("DX", dx);
  ofunc.Done();

  IVLOG(1, to_string(ofunc.prog()));
}

TEST_CASE("ProgGrad", "[deriv]") {
  BoundFunction bf("function (A[I,K], B[K,J]) -> (C) { C[i,j : I,J] = +(A[i,k] * B[k,j]); }");
  Program p = ProgGrad(bf.prog());
}

TEST_CASE("Basic Infeasible Constraints", "[infeasible]") {
  IVLOG(1, "We expect the infeasibility test to throw a warning.");
  Parser p;
  Program prog = p.Parse("function (I[N]) -> (O) {O[2*i] = +(I[4*i]), i - 100 < 50; }");
  Contraction c = prog.ops[0].c;
  REQUIRE_THROWS_AS(Compile(c, {SimpleShape(DataType::FLOAT32, {50}), SimpleShape(DataType::FLOAT32, {50})}),
                    std::runtime_error);
}

TEST_CASE("Duplicate constraints", "[duplicate]") {
  const std::size_t inputSize = 10;
  Parser p;
  Contraction c = p.ParseContraction("O[i] = +(I[i])");

  FlatContraction f =
      Compile(c, {SimpleShape(DataType::FLOAT32, {inputSize}), SimpleShape(DataType::FLOAT32, {inputSize})});
  REQUIRE(f.ranges[0] == inputSize);
  REQUIRE(f.constraints.size() == 0);
}

TEST_CASE("Trivial parallel constraints", "[parallel]") {
  const std::size_t inputSize = 10;
  Parser p;
  Contraction c = p.ParseContraction("O[i] = +(I[2*i])");

  FlatContraction f =
      Compile(c, {SimpleShape(DataType::FLOAT32, {(inputSize + 1) / 2}), SimpleShape(DataType::FLOAT32, {inputSize})});
  REQUIRE(f.ranges[0] == (inputSize + 1) / 2);
  REQUIRE(f.constraints.size() == 0);
}

TEST_CASE("Overlapped parallel constraints", "[parallel]") {
  const std::size_t inputSize = 10;  // minimum 5
  Parser p;
  Contraction c = p.ParseContraction("O[i] = +(I[i - 5])");

  FlatContraction f =
      Compile(c, {SimpleShape(DataType::FLOAT32, {inputSize}), SimpleShape(DataType::FLOAT32, {inputSize})});
  REQUIRE(f.ranges[0] == inputSize - 5);
  REQUIRE(f.constraints.size() == 0);
}

TEST_CASE("Mixed stride parallel constraints", "[parallel]") {
  std::size_t sizeLow = 45;
  std::size_t sizeHi = 55;

  for (std::size_t factor1 = 1; factor1 < 4; ++factor1) {
    for (std::size_t factor2 = 3; factor2 < 6; ++factor2) {
      Parser p;
      std::stringstream progStr;
      progStr << "O[" << std::to_string(factor1) << "*i] = +(I[" << std::to_string(factor2) << "*i])";
      Contraction c = p.ParseContraction(progStr.str());
      for (std::size_t size = sizeLow; size < sizeHi; ++size) {
        FlatContraction f =
            Compile(c, {SimpleShape(DataType::FLOAT32, {size}), SimpleShape(DataType::FLOAT32, {size})});

        IVLOG(2, "Program " << progStr.str());
        IVLOG(3, "Ranges " << f.ranges[0]);
        IVLOG(3, "# Constraints " << f.constraints.size());
        IVLOG(3, "Strides " << f.access[0].strides[0] << ", " << f.access[1].strides[0]);

        REQUIRE(f.ranges[0] == std::min<unsigned int>((size + factor1 - 1) / factor1, (size + factor2 - 1) / factor2));
        REQUIRE(f.constraints.size() == 0);
        REQUIRE(f.access[0].strides[0] == factor1);
        REQUIRE(f.access[1].strides[0] == factor2);
      }
    }
  }
}

TEST_CASE("Parallel constraints with some overlaps that hit nothing", "[parallel]") {
  std::size_t size = 150;
  Parser p;
  Contraction c = p.ParseContraction("O[i/5+j] = +(I[i/3+j,i/2]*J[i/18])");

  FlatContraction f = Compile(c, {SimpleShape(DataType::FLOAT32, {size}), SimpleShape(DataType::FLOAT32, {size, size}),
                                  SimpleShape(DataType::FLOAT32, {size})});

  REQUIRE(f.constraints.size() == 4);
  REQUIRE(f.ranges[0] == 12);
  REQUIRE(f.ranges[1] == 13);
  REQUIRE(f.ranges[2] == 13);
}

TEST_CASE("Parallel constraints 3D", "[parallel]") {
  std::size_t size = 150;
  Parser p;
  Contraction c = p.ParseContraction("O[i/5+j] = +(I[i/3+j,i/2+k,i/18+k])");

  FlatContraction f =
      Compile(c, {SimpleShape(DataType::FLOAT32, {size}), SimpleShape(DataType::FLOAT32, {size, size, size})});

  REQUIRE(f.constraints.size() == 4);
  REQUIRE(f.ranges[0] == 12);
  REQUIRE(f.ranges[1] == 13);
  REQUIRE(f.ranges[2] == 13);
  REQUIRE(f.ranges[3] == 150);
}

TEST_CASE("Non-Parallel constraints 3D", "[parallel]") {
  std::size_t size = 150;
  Parser p;
  Contraction c = p.ParseContraction("O[i/5+j] = +(I[i/3+j,i/2+k,i/18-k])");

  FlatContraction f =
      Compile(c, {SimpleShape(DataType::FLOAT32, {size}), SimpleShape(DataType::FLOAT32, {size, size, size})});

  REQUIRE(f.constraints.size() == 4);
  REQUIRE(f.ranges[0] == 12);
  REQUIRE(f.ranges[1] == 13);
  REQUIRE(f.ranges[2] == 13);
  REQUIRE(f.ranges[3] == 150);
}

TEST_CASE("Condense size 1 dimensions", "[reshape]") {
  std::vector<unsigned int> input_size{77, 1, 2, 3};  // The REQUIREs expect: (High, 1, Lo, Mid)
  Parser p;
  Contraction c = p.ParseContraction("O[n0, n2, n3] = +(I[n0, n1, n2, n3])");

  FlatContraction f =
      Compile(c, {SimpleShape(DataType::FLOAT32, {input_size[0], input_size[2], input_size[3]}),
                  SimpleShape(DataType::FLOAT32, {input_size[0], input_size[1], input_size[2], input_size[3]})});
  REQUIRE(f.ranges[0] == input_size[0]);
  REQUIRE(f.ranges[1] == input_size[2]);
  REQUIRE(f.ranges[2] == input_size[3]);
}

TEST_CASE("Check for switch optimization", "[switch]") {
  auto func = std::make_shared<BoundFunction>(R"***(
      function (T, X[I, J], Y[I, J]) -> (O) {
        O1[] = +(X[i, j] * Y[j, i]);
        O2[] = +(X[i, j] * Y[i, j]);
        O = T ? O1 : O2; 
      }
  )***");

  auto T = FConstValue::make(1.0);
  auto X = TensorValue::make(std::make_shared<BufferBase>(), SimpleShape(DataType::FLOAT32, {10, 10}));
  auto Y = TensorValue::make(std::make_shared<BufferBase>(), SimpleShape(DataType::FLOAT32, {10, 10}));
  auto O = TensorValue::make(std::make_shared<BufferBase>(), SimpleShape(DataType::FLOAT32, {}));

  FunctionApplication a(func);
  a.SetInput("T", T);
  a.SetInput("X", X);
  a.SetInput("Y", Y);

  auto OO = a.GetOutput("O");

  BoundFunction f;
  f.AddDependency(a);
  f.AddUpdate(O, OO);

  RunInfo r = f.PrepareToRun("test");
  IVLOG(1, "New Code:\n" << r.code);
  Parser parser;
  Program prog = parser.Parse(r.code);
  TileOptimizer optimizer;
  KernelList kl = GenerateProgram(prog, r.input_shapes, r.output_shapes, TestGPU(), optimizer);

  assert(kl.kernels.size() == 1);
}

TEST_CASE("Check attribute parsing", "[attr]") {
  Parser p;
  Program prog = p.Parse("function (A[I,K], B[K,J]) -> (O) { [[hello(world)]] O[i,j : I,J] = +(A[i,k] * B[k,j]); }");
  REQUIRE(prog.ops.size() == 1);
  const auto& op = prog.ops[0];
  REQUIRE(op.attributes.size() == 1);
  const auto& attr = op.attributes[0];
  REQUIRE(attr.name() == "hello");
  REQUIRE(attr.params_size() == 1);
  REQUIRE(attr.params().Get(0) == "world");
  REQUIRE(to_string(attr) == "hello(world)");
  auto opstr = to_string(op);
  REQUIRE(opstr.find("[[hello(world)]] O[") == 0);
}

TEST_CASE("CombineConvolutionAndRelu", "[emit]") {
  Parser parser;
  Program prog = parser.Parse(
      "function (B[X,Y], C[Y,Z]) -> (A) { "
      "  T[x,z:X,Z] = +(B[x,y] * C[y,z]); "
      "  M = (T < 0 ? 0.3 * T : T); "
      "  A = (M < 0.9 ? M : 0.9); "
      "}");
  ShapeMap inputs;
  inputs.emplace("B", SimpleShape(DataType::FLOAT32, {10, 10}));
  inputs.emplace("C", SimpleShape(DataType::FLOAT32, {10, 10}));
  ShapeMap outputs;
  outputs.emplace("A", SimpleShape(DataType::FLOAT32, {10, 10}));
  TileOptimizer optimizer;
  auto klist = GenerateProgram(prog, inputs, outputs, TestGPU(), optimizer, "ID");
  if (VLOG_IS_ON(1)) {
    for (const auto& kinfo : klist.kernels) {
      sem::Print emit(*kinfo.kfunc);
      VLOG(1) << "Got kernel: " << emit.str();
    }
  }
  REQUIRE(klist.kernels.size() == 1);
}

TEST_CASE("Tupleism", "[tuple]") {
  Parser parser;
  Program prog = parser.Parse(R"***(
    function (A[I, K], B[K, J]) -> (O) {
      T = tuple(A, B);
      C = element(T, 0);
      D = element(T, 1);
      O[i, j : I, J] = +(C[i, k] * D[k, j]);
    } 
  )***");
  ShapeMap inputs;
  inputs.emplace("A", SimpleShape(DataType::FLOAT32, {10, 17}));
  inputs.emplace("B", SimpleShape(DataType::FLOAT32, {17, 10}));
  ShapeMap outputs;
  outputs.emplace("O", SimpleShape(DataType::FLOAT32, {10, 10}));
  TileOptimizer optimizer;
  auto klist = GenerateProgram(prog, inputs, outputs, TestGPU(), optimizer, "ID");
  if (VLOG_IS_ON(1)) {
    for (const auto& kinfo : klist.kernels) {
      sem::Print emit(*kinfo.kfunc);
      VLOG(1) << "Got kernel: " << emit.str();
    }
  }
  REQUIRE(klist.kernels.size() == 1);
}

}  // namespace
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
