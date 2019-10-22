// Copyright 2017-2018 Intel Corporation.

#include "tile/base/platform_test.h"

#include <gmock/gmock.h>
#include <half.hpp>

#include "base/util/error.h"
#include "base/util/logging.h"
#include "testing/matchers.h"
#include "tile/proto/support.h"

using ::testing::Eq;
using ::testing::NotNull;

namespace pb = google::protobuf;

namespace vertexai {
namespace tile {
namespace testing {

void PlatformTest::SetUp() {
  auto param = GetParam();
  param_ = param.param;
  platform_ = param.factory();
}

std::shared_ptr<Program> PlatformTest::MakeProgram(tile::proto::TileScanningParameters* params,  //
                                                   const char* code,                             //
                                                   const TensorShape& shape) {
  proto::Program pb_program;
  pb_program.set_code(code);
  if (params) {
    *pb_program.mutable_tile_scanning_params() = *params;
  }
  auto pb_shape = IntoProto(shape);
  *(*pb_program.mutable_inputs())["A"].mutable_shape() = pb_shape;
  *(*pb_program.mutable_inputs())["B"].mutable_shape() = pb_shape;
  *(*pb_program.mutable_outputs())["C"].mutable_shape() = pb_shape;
  ConstBufferManager cbm;
  auto program = platform_->MakeProgram(ctx_, pb_program, &cbm);

  EXPECT_THAT(program, NotNull());
  return program;
}

std::vector<int> CastOutput(const TensorShape& shape, View* view) {
  std::vector<int> into;
  switch (shape.type) {
    case DataType::INT8: {
      auto ptr = reinterpret_cast<int8_t*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    case DataType::INT16: {
      auto ptr = reinterpret_cast<int16_t*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    case DataType::INT32: {
      auto ptr = reinterpret_cast<int32_t*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    case DataType::INT64: {
      auto ptr = reinterpret_cast<int64_t*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    case DataType::UINT8: {
      auto ptr = reinterpret_cast<uint8_t*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    case DataType::UINT16: {
      auto ptr = reinterpret_cast<uint16_t*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    case DataType::UINT32: {
      auto ptr = reinterpret_cast<uint32_t*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    case DataType::UINT64: {
      auto ptr = reinterpret_cast<uint64_t*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    case DataType::FLOAT16: {
      auto ptr = reinterpret_cast<half_float::half*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    case DataType::FLOAT32: {
      auto ptr = reinterpret_cast<float*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    case DataType::FLOAT64: {
      auto ptr = reinterpret_cast<double*>(view->data());
      std::copy(ptr, ptr + shape.elem_size(), std::back_inserter(into));
      break;
    }
    default:
      throw std::runtime_error("Unsupported dtype");
  }
  return into;
}

std::shared_ptr<Buffer> PlatformTest::MakeInput(const TensorShape& shape,  //
                                                const std::vector<int>& data) {
  auto buf = platform_->MakeBuffer(ctx_, "", shape.byte_size());
  EXPECT_THAT(buf, NotNull());

  auto view = buf->MapDiscard(ctx_);
  switch (shape.type) {
    case DataType::INT8:
      std::copy(data.begin(), data.end(), reinterpret_cast<int8_t*>(view->data()));
      break;
    case DataType::INT16:
      std::copy(data.begin(), data.end(), reinterpret_cast<int16_t*>(view->data()));
      break;
    case DataType::INT32:
      std::copy(data.begin(), data.end(), reinterpret_cast<int32_t*>(view->data()));
      break;
    case DataType::INT64:
      std::copy(data.begin(), data.end(), reinterpret_cast<int64_t*>(view->data()));
      break;
    case DataType::UINT8:
      std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t*>(view->data()));
      break;
    case DataType::UINT16:
      std::copy(data.begin(), data.end(), reinterpret_cast<uint16_t*>(view->data()));
      break;
    case DataType::UINT32:
      std::copy(data.begin(), data.end(), reinterpret_cast<uint32_t*>(view->data()));
      break;
    case DataType::UINT64:
      std::copy(data.begin(), data.end(), reinterpret_cast<uint64_t*>(view->data()));
      break;
    case DataType::FLOAT16:
      std::copy(data.begin(), data.end(), reinterpret_cast<half_float::half*>(view->data()));
      break;
    case DataType::FLOAT32:
      std::copy(data.begin(), data.end(), reinterpret_cast<float*>(view->data()));
      break;
    case DataType::FLOAT64:
      std::copy(data.begin(), data.end(), reinterpret_cast<double*>(view->data()));
      break;
    default:
      throw std::runtime_error("Unsupported dtype");
  }
  view->WriteBack(ctx_);

  std::vector<int> read_result = CastOutput(shape, buf->MapCurrent(ctx_).get().get());
  EXPECT_THAT(read_result, Eq(data));

  return buf;
}

std::shared_ptr<Buffer> PlatformTest::MakeOutput(const TensorShape& shape) {
  auto buf = platform_->MakeBuffer(ctx_, "", shape.byte_size());
  EXPECT_THAT(buf, NotNull());
  return buf;
}

void PlatformTest::CheckExpected(const TensorShape& shape,            //
                                 const std::shared_ptr<Buffer>& buf,  //
                                 const std::vector<int>& expected) {
  auto view = buf->MapCurrent(ctx_).get();
  std::vector<int> actual = CastOutput(shape, view.get());
  EXPECT_THAT(actual, Eq(expected));
}

namespace multiply {

const char* Code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
// const char* Shape = R"(type: FLOAT32 dims: { size: 4 stride: 4 } dims: { size: 4 stride: 1 })";
const std::vector<int> Input = {
    0, 1, 2, 3,  //
    4, 5, 6, 7,  //
    0, 1, 2, 3,  //
    4, 5, 6, 7,  //
};
const std::vector<int> Expected = {
    16, 22, 28, 34,   //
    48, 70, 92, 114,  //
    16, 22, 28, 34,   //
    48, 70, 92, 114,  //
};

}  // namespace multiply

namespace vector_add {

const char* Code = "function (A, B) -> (C) { C = A + B; }";
// const char* Shape = R"(type: FLOAT32 dims: { size: 4 stride: 4 } dims: { size: 4 stride: 1 })";
const std::vector<int> A = {
    1, 2, 3, 4,  //
    1, 2, 3, 4,  //
    1, 2, 3, 4,  //
    1, 2, 3, 4,  //
};
const std::vector<int> B = {
    5, 6, 7, 8,  //
    5, 6, 7, 8,  //
    5, 6, 7, 8,  //
    5, 6, 7, 8,  //
};
// const std::vector<int> Output = {
//     0, 0, 0, 0,  //
//     0, 0, 0, 0,  //
//     0, 0, 0, 0,  //
//     0, 0, 0, 0,  //
// };
const std::vector<int> Expected = {
    6,   // 1 + 5 = 6
    8,   // 2 + 6 = 8
    10,  // 3 + 7 = 10
    12,  // 4 + 8 = 12
    6,   // 1 + 5 = 6
    8,   // 2 + 6 = 8
    10,  // 3 + 7 = 10
    12,  // 4 + 8 = 12
    6,   // 1 + 5 = 6
    8,   // 2 + 6 = 8
    10,  // 3 + 7 = 10
    12,  // 4 + 8 = 12
    6,   // 1 + 5 = 6
    8,   // 2 + 6 = 8
    10,  // 3 + 7 = 10
    12,  // 4 + 8 = 12
};

}  // namespace vector_add

TEST_P(PlatformTest, VectorAddWorks) {
  auto shape = SimpleShape(param_.dtype, {4, 4});
  auto program = MakeProgram(nullptr, vector_add::Code, shape);
  auto a = MakeInput(shape, vector_add::A);
  auto b = MakeInput(shape, vector_add::B);
  auto c = MakeOutput(shape);
  program->Run(ctx_, {{"A", a}, {"B", b}}, {{"C", c}}).get();
  CheckExpected(shape, c, vector_add::Expected);
}

TEST_P(PlatformTest, MatMulWorks) {
  auto shape = SimpleShape(param_.dtype, {4, 4});
  auto program = MakeProgram(nullptr, multiply::Code, shape);
  auto a = MakeInput(shape, multiply::Input);
  auto b = MakeInput(shape, multiply::Input);
  auto c = MakeOutput(shape);
  program->Run(ctx_, {{"A", a}, {"B", b}}, {{"C", c}}).get();
  CheckExpected(shape, c, multiply::Expected);
}

TEST_P(PlatformTest, RuntimeTileScannerWorks) {
  tile::proto::TileScanningParameters params;
  params.set_max_trials(2);
  params.set_max_trial_runs(2);
  auto shape = SimpleShape(param_.dtype, {4, 4});
  auto program = MakeProgram(&params, multiply::Code, shape);
  auto a = MakeInput(shape, multiply::Input);
  auto b = MakeInput(shape, multiply::Input);
  auto c = MakeOutput(shape);
  program->Run(ctx_, {{"A", a}, {"B", b}}, {{"C", c}}).get();
  CheckExpected(shape, c, multiply::Expected);
}

}  // namespace testing
}  // namespace tile
}  // namespace vertexai
