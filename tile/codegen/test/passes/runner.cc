// Copyright 2019, Intel Corporation

#include "tile/codegen/test/passes/runner.h"

#include <string>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "base/util/file.h"
#include "base/util/logging.h"
#include "tile/codegen/test/passes/manager.h"
#include "tile/stripe/stripe.h"
#include "tile/targets/cpu/jit.h"

namespace fs = boost::filesystem;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {
namespace passes {

std::vector<fs::path> LoadPasses(const fs::path& passes_dir) {
  std::vector<fs::path> ret;
  for (const auto& dir : boost::make_iterator_range(fs::directory_iterator(passes_dir), {})) {
    if (fs::is_regular_file(dir.status()) && dir.path().extension() == ".pb") {
      ret.emplace_back(dir.path());
    }
  }
  std::sort(ret.begin(), ret.end());
  return ret;
}

void RunBaseline(const stripe::Block& entry, BufferManager* inputs, BufferManager* outputs) {
  // Generates fake input values for each top-level buffer in `entry` (stored in `inputs`)
  // Stores the values in each buffer after running the Stripe code in `outputs`
  for (const auto& ref : entry.refs) {
    inputs->add_random(ref.into(), ref.interior_shape.type, ref.interior_shape.elem_size());
  }
  IVLOG(4, "Copying test data");
  *outputs = *inputs;
  auto raw_data = outputs->map_buffers();
  IVLOG(4, "Executing initial Stripe code with CPU Jit");
  targets::cpu::JitExecute(entry, raw_data);
}

void CompareToBaseline(const stripe::Block& entry,       //
                       const BufferManager& inputs,      //
                       BufferManager* expected_outputs,  //
                       const std::string& name) {
  IVLOG(2, "Checking consistency after pass...");
  BufferManager working_buffers(inputs);
  auto raw_data = working_buffers.map_buffers();
  IVLOG(3, "Executing current Stripe code with CPU Jit");
  targets::cpu::JitExecute(entry, raw_data);
  IVLOG(3, "Confirming network output unchanged since previous pass...");
  working_buffers.is_close(*expected_outputs);
}

bool VerifyPasses(const boost::filesystem::path& passes_dir) {
  BufferManager test_inputs;
  BufferManager expected_outputs;
  auto passes = LoadPasses(passes_dir);
  if (passes.empty()) {
    LOG(WARNING) << "No passes detected.";
    return true;
  }
  IVLOG(1, "Verifying passes...");
  for (size_t i = 0; i < passes.size(); i++) {
    const auto& pass = passes[i];
    auto name = pass.stem().string();
    IVLOG(1, name);
    stripe::proto::Program program;
    program.ParseFromString(ReadFile(pass, true));
    auto entry = stripe::FromProto(program)->entry;
    if (i == 0) {
      RunBaseline(*entry, &test_inputs, &expected_outputs);
    } else {
      CompareToBaseline(*entry, test_inputs, &expected_outputs, name);
    }
  }
  return true;
}

}  // namespace passes
}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
