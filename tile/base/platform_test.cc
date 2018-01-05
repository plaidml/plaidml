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

const char* Code = "function (A[X, Y], B[Y, X]) -> (C) { C[x, y : X, Y] = +(A[x, y] * B[y, x]); }";
const char* Shape = R"(type: INT16 dimensions: { size: 4 stride: 4 } dimensions: { size: 4 stride: 1 })";
const uint8_t Input[] = {
    0x00, 0x00,  // [0, 0]
    0x01, 0x00,  // [0, 1]
    0x02, 0x00,  // [0, 2]
    0x03, 0x00,  // [0, 3]
    0x04, 0x00,  // [1, 0]
    0x05, 0x00,  // [1, 1]
    0x06, 0x00,  // [1, 2]
    0x07, 0x00,  // [1, 3]
    0x08, 0x00,  // [2, 0]
    0x09, 0x00,  // [2, 1]
    0x0A, 0x00,  // [2, 2]
    0x0B, 0x00,  // [2, 3]
    0x0C, 0x00,  // [3, 0]
    0x0D, 0x00,  // [3, 1]
    0x0E, 0x00,  // [3, 2]
    0x0F, 0x00,  // [3, 3]
};
const uint8_t Output[] = {
    0x00, 0x00,  // [0, 0]
    0x00, 0x00,  // [0, 1]
    0x00, 0x00,  // [0, 2]
    0x00, 0x00,  // [0, 3]
    0x00, 0x00,  // [1, 0]
    0x00, 0x00,  // [1, 1]
    0x00, 0x00,  // [1, 2]
    0x00, 0x00,  // [1, 3]
    0x00, 0x00,  // [2, 0]
    0x00, 0x00,  // [2, 1]
    0x00, 0x00,  // [2, 2]
    0x00, 0x00,  // [2, 3]
    0x00, 0x00,  // [3, 0]
    0x00, 0x00,  // [3, 1]
    0x00, 0x00,  // [3, 2]
    0x00, 0x00,  // [3, 3]
};
const uint8_t Expected[] = {
    0x00, 0x00,  // [0, 0]
    0x04, 0x00,  // [0, 1]
    0x10, 0x00,  // [0, 2]
    0x24, 0x00,  // [0, 3]
    0x04, 0x00,  // [1, 0]
    0x19, 0x00,  // [1, 1]
    0x36, 0x00,  // [1, 2]
    0x5B, 0x00,  // [1, 3]
    0x10, 0x00,  // [2, 0]
    0x36, 0x00,  // [2, 1]
    0x64, 0x00,  // [2, 2]
    0x9A, 0x00,  // [2, 3]
    0x24, 0x00,  // [3, 0]
    0x5B, 0x00,  // [3, 1]
    0x9A, 0x00,  // [3, 2]
    0xE1, 0x00,  // [3, 3]
};

}  // namespace multiply

namespace vector_add {

const char* Code = "function (A, B) -> (C) { C = A + B; }";
const char* Shape = R"(type: INT16 dimensions: { size: 4 stride: 4 } dimensions: { size: 4 stride: 1 })";
const uint8_t A[] = {
    0x01, 0x00,  // 1
    0x02, 0x00,  // 2
    0x03, 0x00,  // 3
    0x04, 0x00,  // 4
    0x01, 0x00,  // 1
    0x02, 0x00,  // 2
    0x03, 0x00,  // 3
    0x04, 0x00,  // 4
    0x01, 0x00,  // 1
    0x02, 0x00,  // 2
    0x03, 0x00,  // 3
    0x04, 0x00,  // 4
    0x01, 0x00,  // 1
    0x02, 0x00,  // 2
    0x03, 0x00,  // 3
    0x04, 0x00,  // 4
};
const uint8_t B[] = {
    0x05, 0x00,  // 5
    0x06, 0x00,  // 6
    0x07, 0x00,  // 7
    0x08, 0x00,  // 8
    0x05, 0x00,  // 5
    0x06, 0x00,  // 6
    0x07, 0x00,  // 7
    0x08, 0x00,  // 8
    0x05, 0x00,  // 5
    0x06, 0x00,  // 6
    0x07, 0x00,  // 7
    0x08, 0x00,  // 8
    0x05, 0x00,  // 5
    0x06, 0x00,  // 6
    0x07, 0x00,  // 7
    0x08, 0x00,  // 8
};
const uint8_t Output[] = {
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
    0x00, 0x00,  // 0
};
const uint8_t Expected[] = {
    0x06, 0x00,  // 1 + 5 = 6
    0x08, 0x00,  // 2 + 6 = 8
    0x0A, 0x00,  // 3 + 7 = 10
    0x0C, 0x00,  // 4 + 8 = 12
    0x06, 0x00,  // 1 + 5 = 6
    0x08, 0x00,  // 2 + 6 = 8
    0x0A, 0x00,  // 3 + 7 = 10
    0x0C, 0x00,  // 4 + 8 = 12
    0x06, 0x00,  // 1 + 5 = 6
    0x08, 0x00,  // 2 + 6 = 8
    0x0A, 0x00,  // 3 + 7 = 10
    0x0C, 0x00,  // 4 + 8 = 12
    0x06, 0x00,  // 1 + 5 = 6
    0x08, 0x00,  // 2 + 6 = 8
    0x0A, 0x00,  // 3 + 7 = 10
    0x0C, 0x00,  // 4 + 8 = 12
};

}  // namespace vector_add

std::string MakeBuffer(const uint8_t* data, size_t len) {
  return std::string(reinterpret_cast<const char*>(data), len);
}

TEST_P(PlatformTest, VectorAddWorks) {
  context::Context ctx;
  auto device = MakePlatform();
  auto program = MakeProgram(ctx, device, nullptr, vector_add::Code, vector_add::Shape);
  auto a = MakeInput(ctx, device, MakeBuffer(vector_add::A, sizeof(vector_add::A)));
  auto b = MakeInput(ctx, device, MakeBuffer(vector_add::B, sizeof(vector_add::B)));
  auto c = MakeOutput(ctx, device, MakeBuffer(vector_add::Output, sizeof(vector_add::Output)));
  program->Run(ctx, {{"A", a}, {"B", b}}, {{"C", c}});
  CheckExpected(ctx, c, MakeBuffer(vector_add::Expected, sizeof(vector_add::Expected)));
}

TEST_P(PlatformTest, PiecewiseMultiplyWorks) {
  context::Context ctx;
  auto device = MakePlatform();
  auto program = MakeProgram(ctx, device, nullptr, multiply::Code, multiply::Shape);
  auto a = MakeInput(ctx, device, MakeBuffer(multiply::Input, sizeof(multiply::Input)));
  auto b = MakeInput(ctx, device, MakeBuffer(multiply::Input, sizeof(multiply::Input)));
  auto c = MakeOutput(ctx, device, MakeBuffer(multiply::Output, sizeof(multiply::Output)));
  program->Run(ctx, {{"A", a}, {"B", b}}, {{"C", c}});
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
  program->Run(ctx, {{"A", a}, {"B", b}}, {{"C", c}});
  CheckExpected(ctx, c, MakeBuffer(multiply::Expected, sizeof(multiply::Expected)));
}

}  // namespace testing
}  // namespace tile
}  // namespace vertexai
