// Copyright 2017, Vertex.AI.

#pragma once

#include <functional>
#include <memory>

#include <gtest/gtest.h>

#include "base/util/compat.h"
#include "tile/base/platform.h"

namespace vertexai {
namespace tile {
namespace testing {

typedef std::function<std::unique_ptr<tile::Platform>()> PlatformFactory;

// Platform implementation conformance tests.
//
// To test a platform, #include this header (linking with the :platform_tests
// target), and use INSTANTIATE_TEST_CASE_P to instantiate the conformance
// tests with factories producing Platform instances -- e.g.
//
//   INSTANTIATE_TEST_CASE_P(
//       MyPlatform,
//       PlatformTest,
//       ::testing::Values(std::function<unique_ptr<Platform>()>[](){
//         return compat::make_unique<MyPlatform>();
//       }));
//
class PlatformTest : public ::testing::TestWithParam<PlatformFactory> {
 protected:
  std::unique_ptr<tile::Platform> MakePlatform();
};

namespace multiply {

// The following functions can be used to build up a simple matrix multiplication test.
// All buffers in this code have the same shape: INT16, dim{size=4, stride=8}, dim{size=4, stride=2}.

// Make a program: C[x, y] = +(A[x, y] * B[y, x]).
std::unique_ptr<Program> MakeProgram(const context::Context& ctx, const std::unique_ptr<Platform>& device,
                                     tile::proto::TileScanningParameters* params = nullptr);

// Create A, the first input.
std::shared_ptr<Buffer> MakeInA(const context::Context& ctx, const std::unique_ptr<Platform>& device);

// Create B, the second input.
std::shared_ptr<Buffer> MakeInB(const context::Context& ctx, const std::unique_ptr<Platform>& device);

// Create C, the output buffer.
std::shared_ptr<Buffer> MakeOutC(const context::Context& ctx, const std::unique_ptr<Platform>& device);

// Run the program with the specified inputs and outputs.
void Run(const context::Context& ctx, const std::unique_ptr<Program>& program, const std::shared_ptr<Buffer>& a,
         const std::shared_ptr<Buffer>& b, const std::shared_ptr<Buffer>& c);

// Validate the final contents of C.
void CheckExpected(const context::Context& ctx, const std::shared_ptr<Buffer>& c);

}  // namespace multiply
}  // namespace testing
}  // namespace tile
}  // namespace vertexai
