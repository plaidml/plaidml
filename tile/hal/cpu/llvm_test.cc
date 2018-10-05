// Copyright 2017-2018 Intel Corporation.

#include <gmock/gmock.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>

#include <half.hpp>

#include "tile/hal/cpu/emitllvm.h"
#include "tile/hal/cpu/runtime.h"
#include "tile/lang/sembuilder.h"
#include "tile/lang/semtree.h"

using ::testing::Eq;
using ::testing::Ne;
using ::testing::NotNull;

namespace half_float {
void PrintTo(const half h, ::std::ostream* os) { *os << static_cast<float>(h); }
}  // namespace half_float

namespace vertexai {
namespace tile {
namespace testing {
namespace {

static const sem::Type idxType{sem::Type::INDEX};
static const sem::Type voidType{sem::Type::TVOID};
static const sem::Type int16Type{sem::Type::VALUE, DataType::INT16};
static const sem::Type int32Type{sem::Type::VALUE, DataType::INT32};
static const sem::Type int64Type{sem::Type::VALUE, DataType::INT64};
static const sem::Type fp32Type{sem::Type::VALUE, DataType::FLOAT32};
static const sem::Type fp64Type{sem::Type::VALUE, DataType::FLOAT64};
static const sem::Type ptrInt32Type{sem::Type::POINTER_MUT, DataType::INT32};
static const sem::Type ptrFP32Type{sem::Type::POINTER_MUT, DataType::FLOAT32};

static llvm::ExecutionEngine* JIT(const sem::Node& n) {
  tile::hal::cpu::Emit emit;
  n.Accept(emit);
  std::string errStr;
  std::unique_ptr<llvm::RuntimeDyld::SymbolResolver> rez(new tile::hal::cpu::Runtime);
  LLVMInitializeNativeTarget();
  LLVMLinkInMCJIT();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();
  llvm::ExecutionEngine* engine = llvm::EngineBuilder(std::move(emit.result()))
                                      .setErrorStr(&errStr)
                                      .setEngineKind(llvm::EngineKind::JIT)
                                      .setVerifyModules(true)
                                      .setSymbolResolver(std::move(rez))
                                      .create();
  if (engine) {
    engine->finalizeObject();
  } else {
    std::cerr << "Failed to create ExecutionEngine: " << errStr << std::endl;
  }
  return engine;
}

TEST(CpuDevice, LLVM_minimal) {
  using namespace sem::builder;  // NOLINT
  auto i = _("i");
  auto f = _Function("passthrough", idxType, {{idxType, "i"}}, {_Return(i)});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (ssize_t(*)(ssize_t))engine->getFunctionAddress("passthrough");
  EXPECT_THAT(e, NotNull());
  EXPECT_THAT(e(42), Eq(42));
}

TEST(CpuDevice, LLVM_factorial) {
  using namespace sem::builder;  // NOLINT
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

  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (ssize_t(*)(ssize_t))engine->getFunctionAddress("factorial");
  EXPECT_THAT(e, NotNull());
  EXPECT_THAT(e(3), Eq(6));
  EXPECT_THAT(e(4), Eq(24));
}

TEST(CpuDevice, LLVM_indirect_assign) {
  using namespace sem::builder;  // NOLINT
  auto f = _Function("copybot", voidType, {{ptrInt32Type, "a"}, {ptrInt32Type, "b"}},
                     {_("a")[_Const(0)] = _("b")[_Const(0)]});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (void (*)(int32_t*, int32_t*))engine->getFunctionAddress("copybot");
  int32_t a = 19;
  int32_t b = 42;
  e(&a, &b);
  EXPECT_THAT(a, 42);
  EXPECT_THAT(b, 42);
}

TEST(CpuDevice, LLVM_forloop) {
  using namespace sem::builder;  // NOLINT
  auto i = _("i");
  auto r = _("r");
  auto f = _Function("floop", idxType, {{idxType, "z"}},
                     {_Declare(idxType, "r", _Const(1)), _For("i", 10, 1, r = r + i), _Return(r)});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (ssize_t(*)(ssize_t))engine->getFunctionAddress("floop");
  EXPECT_THAT(e, NotNull());
  EXPECT_THAT(e(0), Eq(46));
}

TEST(CpuDevice, LLVM_cond) {
  using namespace sem::builder;  // NOLINT
  auto z = _("z");
  auto r = _("r");
  auto f = _Function("cond", idxType, {{idxType, "z"}, {idxType, "r"}}, {_Return(_Cond(z > _Const(42), z + r, z - r))});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (ssize_t(*)(ssize_t, ssize_t))engine->getFunctionAddress("cond");
  EXPECT_THAT(e, NotNull());
  EXPECT_THAT(e(0, 10), Eq(-10));
  EXPECT_THAT(e(100, 10), Eq(110));
  EXPECT_THAT(e(42, 10), Eq(32));
  EXPECT_THAT(e(50, 10), Eq(60));
}

TEST(CpuDevice, LLVM_select) {
  using namespace sem::builder;  // NOLINT
  auto z = _("z");
  auto r = _("r");
  auto f =
      _Function("select", idxType, {{idxType, "z"}, {idxType, "r"}}, {_Return(_Select(z > _Const(42), z + r, z - r))});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (ssize_t(*)(ssize_t, ssize_t))engine->getFunctionAddress("select");
  EXPECT_THAT(e, NotNull());
  EXPECT_THAT(e(0, 10), Eq(-10));
  EXPECT_THAT(e(100, 10), Eq(110));
  EXPECT_THAT(e(42, 10), Eq(32));
  EXPECT_THAT(e(50, 10), Eq(60));
}

TEST(CpuDevice, LLVM_arrayset) {
  using namespace sem::builder;  // NOLINT
  auto buf = _("buf");
  auto n = _("n");
  auto step = _("step");
  auto i = _("i");
  auto f =
      _Function("arrayset", voidType, {{ptrInt32Type, "buf"}, {idxType, "n"}, {idxType, "step"}},
                {_Declare(idxType, "i", _Const(0)), _While(i < n, _Block({buf[i] = i * step, i = i + _Const(1)}))});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (void (*)(int32_t*, ssize_t, ssize_t))engine->getFunctionAddress("arrayset");
  EXPECT_THAT(e, NotNull());
  int32_t data[6]{123, 123, 123, 123, 123, 123};
  e(data, 4, 1);
  EXPECT_THAT(data[0], Eq(0));
  EXPECT_THAT(data[1], Eq(1));
  EXPECT_THAT(data[2], Eq(2));
  EXPECT_THAT(data[3], Eq(3));
  EXPECT_THAT(data[4], Eq(123));
  e(data, 5, 2);
  EXPECT_THAT(data[0], Eq(0));
  EXPECT_THAT(data[1], Eq(2));
  EXPECT_THAT(data[2], Eq(4));
  EXPECT_THAT(data[3], Eq(6));
  EXPECT_THAT(data[4], Eq(8));
  EXPECT_THAT(data[5], Eq(123));
  e(data, 3, 8);
  EXPECT_THAT(data[0], Eq(0));
  EXPECT_THAT(data[1], Eq(8));
  EXPECT_THAT(data[2], Eq(16));
  EXPECT_THAT(data[3], Eq(6));
  EXPECT_THAT(data[4], Eq(8));
  EXPECT_THAT(data[5], Eq(123));
}

TEST(CpuDevice, LLVM_fp_add) {
  using namespace sem::builder;  // NOLINT
  auto a = _("a");
  auto b = _("b");
  auto out = _("out");
  auto i = _("i");
  const size_t arraylen = 4;
  auto f = _Function("fp_add", voidType,
                     {{
                          ptrFP32Type,
                          "a",
                      },
                      {ptrFP32Type, "b"},
                      {ptrFP32Type, "out"}},
                     {_For("i", arraylen, 1, out[i] = a[i] + b[i])});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (void (*)(float*, float*, float*))engine->getFunctionAddress("fp_add");
  EXPECT_THAT(e, NotNull());
  float buf_a[arraylen]{407.111, 2, 0.9703, -10000000000000000.0};
  float buf_b[arraylen]{-82.510551960968, -84.5980367, 27.26, 1.0};
  float buf_out[arraylen]{0.0, 0.0, 0.0, 0.0};
  e(buf_a, buf_b, buf_out);
  for (size_t i = 0; i < arraylen; ++i) {
    EXPECT_THAT(buf_out[i], Eq(buf_a[i] + buf_b[i]));
  }
}

TEST(CpuDevice, LLVM_mix_mult) {
  using namespace sem::builder;  // NOLINT
  auto a = _("a");
  auto b = _("b");
  auto out = _("out");
  auto i = _("i");
  const size_t arraylen = 4;
  auto f = _Function("fp_add", voidType,
                     {{
                          ptrFP32Type,
                          "a",
                      },
                      {ptrInt32Type, "b"},
                      {ptrFP32Type, "out"}},
                     {_For("i", arraylen, 1, out[i] = a[i] * b[i])});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (void (*)(float*, int32_t*, float*))engine->getFunctionAddress("fp_add");
  EXPECT_THAT(e, NotNull());
  float buf_a[arraylen]{407.111, 2, 0.9703, -10000000000000000.0};
  int32_t buf_b[arraylen]{51, 91, -12, 74};
  float buf_out[arraylen]{0.0, 0.0, 0.0, 0.0};
  e(buf_a, buf_b, buf_out);
  for (size_t i = 0; i < arraylen; ++i) {
    EXPECT_THAT(buf_out[i], Eq(buf_a[i] * buf_b[i]));
  }
}

TEST(CpuDevice, LLVM_array_init) {
  using namespace sem::builder;  // NOLINT
  const size_t arraylen = 4;
  sem::Type fp32arr = {sem::Type::VALUE, DataType::FLOAT32, 1, arraylen};
  auto v = _("v");
  auto a = _("a");
  auto out = _("out");
  auto i = _("i");
  auto f = _Function("array_init", voidType, {{fp32Type, "v"}, {ptrFP32Type, "out"}},
                     {_Declare(fp32arr, "a", v), _For("i", arraylen, 1, out[i] = a[i] + out[i])});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (void (*)(float, float*))engine->getFunctionAddress("array_init");
  EXPECT_THAT(e, NotNull());
  float buf_out[arraylen]{1.0, 2.0, 3.0, 4.0};
  e(100.0, buf_out);
  float expect[arraylen]{101.0, 102.0, 103.0, 104.0};
  for (size_t i = 0; i < arraylen; ++i) {
    EXPECT_THAT(buf_out[i], Eq(expect[i]));
  }
}

TEST(CpuDevice, LLVM_block_scope) {
  using namespace sem::builder;  // NOLINT
  auto var = _("var");
  auto sum = _("sum");
  auto in = _("in");
  auto f = _Function("add3", int32Type, {{int32Type, "in"}},
                     {_Declare(int32Type, "sum", in), _Block({_Declare(int32Type, "var", _Const(1)), sum = sum + var}),
                      _Block({_Declare(int32Type, "var", _Const(2)), sum = sum + var}), _Return(sum)});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (int32_t(*)(int32_t))engine->getFunctionAddress("add3");
  EXPECT_THAT(e, NotNull());
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_THAT(e(i), Eq(i + 3));
  }
}

TEST(CpuDevice, LLVM_assignment_coercion) {
  using namespace sem::builder;  // NOLINT
  auto in16 = _("in16");
  auto temp64 = _("temp64");
  auto tempFP = _("tempFP");
  auto f = _Function("assignthrough", int16Type, {{int16Type, "in16"}},
                     {_Declare(int64Type, "temp64", _Const(0)), temp64 = in16, _Declare(fp32Type, "tempFP", _Const(0)),
                      tempFP = temp64, _Return(tempFP)});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (int16_t(*)(int16_t))engine->getFunctionAddress("assignthrough");
  EXPECT_THAT(e, NotNull());
  for (int32_t i = -2000; i < 4702; ++i) {
    EXPECT_THAT(e(i), Eq(i));
  }
}

TEST(CpuDevice, LLVM_negation) {
  using namespace sem::builder;  // NOLINT
  const size_t arraylen = 6;
  auto buf = _("buf");
  auto i = _("i");
  auto f = _Function("negator", voidType, {{ptrInt32Type, "buf"}}, {_For("i", arraylen, 1, buf[i] = -buf[i])});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (void (*)(int32_t*))engine->getFunctionAddress("negator");
  EXPECT_THAT(e, NotNull());
  int32_t data[arraylen]{123, 456, 789, 101, 10101, 1010101};
  e(data);
  int32_t expect[arraylen]{-123, -456, -789, -101, -10101, -1010101};
  for (unsigned i = 0; i < arraylen; ++i) {
    EXPECT_THAT(data[i], Eq(expect[i]));
  }
}

TEST(CpuDevice, LLVM_mix_casting) {
  using namespace sem::builder;  // NOLINT
  auto in = _("in");
  auto f = _Function("mixcast", fp32Type, {{fp32Type, "in"}},
                     {_Return(_Cast(int32Type, _Cast(int64Type, _Cast(fp64Type, in))))});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto e = (float (*)(float))engine->getFunctionAddress("mixcast");
  EXPECT_THAT(e, NotNull());
  static const size_t arbitrary_len = 8;
  uint32_t data[arbitrary_len]{0x5f683b2d, 0xb90d91f6, 0x6b1b3158, 0x73053a70,
                               0x5fa54307, 0x4d81ff7e, 0x58916ad8, 0x9a9b68dc};
  for (unsigned i = 0; i < arbitrary_len; ++i) {
    union {
      uint32_t source;
      float dest;
    } bitcast;
    bitcast.source = data[i];
    float val = bitcast.dest;
    float expect = static_cast<int32_t>(static_cast<int64_t>(static_cast<double>(val)));
    EXPECT_THAT(e(val), Eq(expect));
  }
}

TEST(CpuDevice, LLVM_rounding_builtins) {
  using namespace sem::builder;  // NOLINT
  std::vector<float> vals{1.7, 0.8, 0.5, 0.9, -0.3, -0.8, 0, 1.7, 0.6};
  std::vector<std::string> funcs{"round", "ceil", "floor"};
  std::map<std::string, std::vector<float>> results{{"round", {2, 1, 1, 1, -0, -1, 0, 2, 1}},
                                                    {"ceil", {2, 1, 1, 1, 0, 0, 0, 2, 1}},
                                                    {"floor", {1, 0, 0, 0, -1, -1, 0, 1, 0}}};
  for (auto builtin : funcs) {
    auto in = _("in");
    auto f = _Function("builtin", fp32Type, {{fp32Type, "in"}}, {_Return(_(builtin)(_("in")))});
    auto engine = JIT(*f);
    EXPECT_THAT(engine, NotNull());
    auto e = (float (*)(float))engine->getFunctionAddress("builtin");
    EXPECT_THAT(e, NotNull());
    auto& res = results[builtin];
    assert(res.size() == vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
      EXPECT_THAT(e(vals[i]), Eq(res[i]));
    }
  }
}

TEST(CpuDevice, LLVM_half_float_vec) {
  using namespace sem::builder;  // NOLINT
  typedef half_float::half half;
  for (size_t vecwidth : std::vector<size_t>{1, 2, 4, 8}) {
    sem::Type type{sem::Type::POINTER_MUT, DataType::FLOAT16, vecwidth};
    auto f = _Function("kernel", voidType, {{type, "a"}, {type, "b"}, {type, "out"}},
                       {_("out")[_Const(0)] = _("out")[_Const(0)] + _("a")[_Const(0)] * _("b")[_Const(0)]});
    auto engine = JIT(*f);
    EXPECT_THAT(engine, NotNull());
    auto kernel = (void (*)(half*, half*, half*))engine->getFunctionAddress("kernel");
    EXPECT_THAT(kernel, NotNull());
    std::vector<half> a(vecwidth, static_cast<half>(2.0f));
    std::vector<half> b(vecwidth, static_cast<half>(3.0f));
    std::vector<half> out(vecwidth, static_cast<half>(1.0f));
    kernel(a.data(), b.data(), out.data());
    for (size_t i = 0; i < vecwidth; ++i) {
      EXPECT_THAT(out[i], Eq(7));
    }
  }
}

TEST(CpuDevice, LLVM_vec_add_loop) {
  using namespace sem::builder;  // NOLINT
  sem::Type float4{sem::Type::VALUE, DataType::FLOAT32, 4};
  sem::Type ptrFloat4{sem::Type::POINTER_MUT, DataType::FLOAT32, 4};
  auto f = _Function(
      "kernel", voidType, {{int32Type, "group_id"}, {ptrFloat4, "C"}, {ptrFloat4, "A"}, {ptrFloat4, "B"}},
      {_Declare(int32Type, "i1_i2_gid", _("group_id") * _Const(2)),
       _For("i1_i2_lid", 2, 1,
            _Block({_Declare(int32Type, "gout_idx", _("i1_i2_gid") + _("i1_i2_lid")),
                    _Declare(float4, "LA", _("A")[_("gout_idx")]), _Declare(float4, "LB", _("B")[_("gout_idx")]),
                    _Declare(float4, "LC", _Cast(float4, _("LA")) + _Cast(float4, _("LB"))),
                    _("C")[_("gout_idx")] = _("LC")}))});
  auto engine = JIT(*f);
  EXPECT_THAT(engine, NotNull());
  auto kernel = (void (*)(int32_t, float*, float*, float*))engine->getFunctionAddress("kernel");
  EXPECT_THAT(kernel, NotNull());
  float A[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  float B[] = {5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
  float Output[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float Expect[] = {6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12};
  for (unsigned group_id = 0; group_id < 2; ++group_id) {
    kernel(group_id, Output, A, B);
  }
  for (unsigned i = 0; i < 16; ++i) {
    EXPECT_THAT(Output[i], Eq(Expect[i]));
  }
}

}  // namespace
}  // namespace testing
}  // namespace tile
}  // namespace vertexai
