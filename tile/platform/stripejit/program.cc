// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/stripejit/program.h"

#include <memory>

#include "base/util/env.h"
#include "tile/codegen/driver.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/parser.h"
#include "tile/proto/support.h"
#include "tile/targets/cpu/jit.h"

namespace vertexai {
namespace tile {
namespace stripejit {

Program::Program(const context::Context& ctx, const tile::proto::Program& program)
    : executable_{new targets::cpu::Native} {
  lang::Parser parser;
  lang::RunInfo runinfo;
  runinfo.program = parser.Parse(program.code());
  runinfo.input_shapes = FromProto(program.inputs());
  runinfo.output_shapes = FromProto(program.outputs());
  runinfo.program_name = "stripe_program";
  auto stripe = GenerateStripe(runinfo);
  auto out_dir = env::Get("STRIPE_OUTPUT");
  codegen::OptimizeOptions options = {
      !out_dir.empty(),     // dump_passes
      false,                // dump_code
      out_dir + "/passes",  // dbg_dir
  };
  auto cfg = codegen::Configs::Resolve("cpu/cpu");
  codegen::Optimize(stripe->entry.get(), cfg.passes(), options);
  executable_->compile(*stripe->entry);
}

Program::~Program() {}

boost::future<void> Program::Run(const context::Context& ctx,
                                 std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                                 std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) {
  std::map<std::string, void*> buffers;
  // map in the input buffers, preserving contents
  for (auto& iter : inputs) {
    std::string name = iter.first;
    auto view = iter.second->MapCurrent(ctx).get();
    buffers.emplace(name, view->data());
  }
  // map in output buffers, discarding contents
  for (auto& iter : outputs) {
    std::string name = iter.first;
    // don't overwrite the buffer if it's already been mapped in
    if (buffers.find(name) == buffers.end()) {
      auto view = iter.second->MapDiscard(ctx);
      buffers.emplace(name, view->data());
    }
  }
  executable_->run(buffers);
  return boost::make_ready_future();
}

}  // namespace stripejit
}  // namespace tile
}  // namespace vertexai
