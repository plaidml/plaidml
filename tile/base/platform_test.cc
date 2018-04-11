#include "tile/base/platform_test.h"

#include <gmock/gmock.h>

#include <google/protobuf/text_format.h>

#include "base/util/error.h"
#include "base/util/logging.h"
#include "testing/matchers.h"

using ::testing::Eq;
using ::testing::EqualsProto;
using ::testing::Ge;
using ::testing::NotNull;
using ::testing::SizeIs;

namespace pb = google::protobuf;

namespace vertexai {
namespace tile {
namespace testing {

std::unique_ptr<Program> MakeProgram(const context::Context& ctx, const std::unique_ptr<Platform>& device,
                                     tile::proto::TileScanningParameters* params, const char* code,
                                     const char* shape_str) {
  proto::TensorShape shape;
  proto::Program pprogram;
  pprogram.set_code(code);
  pb::TextFormat::ParseFromString(shape_str, &shape);
  if (params) {
    *pprogram.mutable_tile_scanning_params() = *params;
  }
  (*pprogram.mutable_inputs())["A"] = shape;
  (*pprogram.mutable_inputs())["B"] = shape;
  (*pprogram.mutable_outputs())["C"] = shape;
  auto program = device->MakeProgram(ctx, pprogram);

  EXPECT_THAT(program, NotNull());
  return program;
}

std::shared_ptr<Buffer> MakeInput(const context::Context& ctx, const std::unique_ptr<Platform>& device,
                                  const std::string& str) {
  auto buf = device->MakeBuffer(ctx, "", str.size());
  EXPECT_THAT(buf, NotNull());

  auto view = buf->MapDiscard(ctx);
  str.copy(reinterpret_cast<char*>(view->data()), view->size());
  view->WriteBack(ctx);

  auto read_result = buf->MapCurrent(ctx).get()->str();
  EXPECT_THAT(read_result, Eq(str));

  return buf;
}

std::shared_ptr<Buffer> MakeOutput(const context::Context& ctx, const std::unique_ptr<Platform>& device,
                                   const std::string& str) {
  auto buf = device->MakeBuffer(ctx, "", str.size());
  EXPECT_THAT(buf, NotNull());

  auto view = buf->MapDiscard(ctx);
  str.copy(reinterpret_cast<char*>(view->data()), view->size());
  view->WriteBack(ctx);

  return buf;
}

void CheckExpected(const context::Context& ctx, const std::shared_ptr<Buffer>& buf, const std::string& expected) {
  auto read_result = buf->MapCurrent(ctx).get()->str();
  EXPECT_THAT(read_result, Eq(expected));
}

std::unique_ptr<tile::Platform> PlatformTest::MakePlatform() { return GetParam()(); }

TEST_P(PlatformTest, ConstructDestruct) {
  auto device = MakePlatform();
  EXPECT_THAT(device, NotNull());
}

TEST_P(PlatformTest, ListDevicesReturnsAtLeastOneDevice) {
  auto device = MakePlatform();

  proto::ListDevicesRequest request;
  proto::ListDevicesResponse response;
  device->ListDevices(context::Context(), request, &response);
  EXPECT_THAT(response.devices(), SizeIs(Ge(1)));
}

namespace multiply {

const char* Code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
const char* Shape = R"(type: FLOAT32 dimensions: { size: 4 stride: 4 } dimensions: { size: 4 stride: 1 })";
const float Input[] = {
    0,  1,  2,  3,   //
    4,  5,  6,  7,   //
    8,  9,  10, 11,  //
    12, 13, 14, 15,  //
};
const float Output[] = {
    0, 0, 0, 0,  //
    0, 0, 0, 0,  //
    0, 0, 0, 0,  //
    0, 0, 0, 0,  //
};
const float Expected[] = {
    56,  62,  68,  74,   //
    152, 174, 196, 218,  //
    248, 286, 324, 362,  //
    344, 398, 452, 506,  //
};

}  // namespace multiply

namespace vector_add {

const char* Code = "function (A, B) -> (C) { C = A + B; }";
const char* Shape = R"(type: FLOAT32 dimensions: { size: 4 stride: 4 } dimensions: { size: 4 stride: 1 })";
const float A[] = {
    1, 2, 3, 4,  //
    1, 2, 3, 4,  //
    1, 2, 3, 4,  //
    1, 2, 3, 4,  //
};
const float B[] = {
    5, 6, 7, 8,  //
    5, 6, 7, 8,  //
    5, 6, 7, 8,  //
    5, 6, 7, 8,  //
};
const float Output[] = {
    0, 0, 0, 0,  //
    0, 0, 0, 0,  //
    0, 0, 0, 0,  //
    0, 0, 0, 0,  //
};
const float Expected[] = {
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

std::string MakeBuffer(const float* data, size_t len) { return std::string(reinterpret_cast<const char*>(data), len); }

TEST_P(PlatformTest, VectorAddWorks) {
  context::Context ctx;
  auto device = MakePlatform();
  auto program = MakeProgram(ctx, device, nullptr, vector_add::Code, vector_add::Shape);
  auto a = MakeInput(ctx, device, MakeBuffer(vector_add::A, sizeof(vector_add::A)));
  auto b = MakeInput(ctx, device, MakeBuffer(vector_add::B, sizeof(vector_add::B)));
  auto c = MakeOutput(ctx, device, MakeBuffer(vector_add::Output, sizeof(vector_add::Output)));
  program->Run(ctx, {{"A", a}, {"B", b}}, {{"C", c}}).get();
  CheckExpected(ctx, c, MakeBuffer(vector_add::Expected, sizeof(vector_add::Expected)));
}

TEST_P(PlatformTest, MatMulWorks) {
  context::Context ctx;
  auto device = MakePlatform();
  auto program = MakeProgram(ctx, device, nullptr, multiply::Code, multiply::Shape);
  auto a = MakeInput(ctx, device, MakeBuffer(multiply::Input, sizeof(multiply::Input)));
  auto b = MakeInput(ctx, device, MakeBuffer(multiply::Input, sizeof(multiply::Input)));
  auto c = MakeOutput(ctx, device, MakeBuffer(multiply::Output, sizeof(multiply::Output)));
  program->Run(ctx, {{"A", a}, {"B", b}}, {{"C", c}}).get();
  CheckExpected(ctx, c, MakeBuffer(multiply::Expected, sizeof(multiply::Expected)));
}

TEST_P(PlatformTest, DISABLED_RuntimeTileScannerWorks) {
  context::Context ctx;
  tile::proto::TileScanningParameters params;
  params.set_max_trials(2);
  params.set_max_trial_runs(2);
  auto device = MakePlatform();
  auto program = MakeProgram(ctx, device, &params, multiply::Code, multiply::Shape);
  auto a = MakeInput(ctx, device, MakeBuffer(multiply::Input, sizeof(multiply::Input)));
  auto b = MakeInput(ctx, device, MakeBuffer(multiply::Input, sizeof(multiply::Input)));
  auto c = MakeOutput(ctx, device, MakeBuffer(multiply::Output, sizeof(multiply::Output)));
  program->Run(ctx, {{"A", a}, {"B", b}}, {{"C", c}}).get();
  CheckExpected(ctx, c, MakeBuffer(multiply::Expected, sizeof(multiply::Expected)));
}

}  // namespace testing
}  // namespace tile
}  // namespace vertexai
