// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/cpu_program.h"

#include <memory>

#include "base/util/env.h"
#include "tile/codegen/driver.h"
#include "tile/lang/gen_stripe.h"
#include "tile/targets/cpu/jit.h"
#include "tile/targets/targets.h"

namespace vertexai {
namespace tile {
namespace local_machine {

CpuProgram::CpuProgram(            //
    const std::string& target,     //
    const lang::RunInfo& runinfo,  //
    ConstBufferManager* const_bufs)
    : executable_{new targets::cpu::Native} {
  auto stripe = GenerateStripe(runinfo);
  auto out_dir = boost::filesystem::path(env::Get("PLAIDML_STRIPE_OUTPUT"));
  codegen::OptimizeOptions options = {
      !out_dir.empty(),    // dump_passes
      false,               // dump_passes_proto
      false,               // dump_code
      out_dir / "passes",  // dbg_dir
  };
  const auto& cfgs = targets::GetConfigs();
  const auto& cfg = cfgs.configs().at(target);
  const auto& stage = cfg.stages().at("default");
  codegen::CompilerState state(stripe);
  state.const_bufs = const_bufs;
  codegen::Optimize(&state, stage.passes(), options);
  targets::cpu::Config config;
  if (!env::Get("PLAIDML_CPU_PROFILE").empty()) {
    config.profile_block_execution = true;
    source_ = stripe->entry;
  }
  executable_->compile(*stripe->entry, config);
}

CpuProgram::CpuProgram(                              //
    const std::string& target,                       //
    const std::shared_ptr<stripe::Program>& stripe,  //
    ConstBufferManager* const_bufs)
    : executable_{new targets::cpu::Native} {
  auto out_dir = boost::filesystem::path(env::Get("PLAIDML_STRIPE_OUTPUT"));
  codegen::OptimizeOptions options = {
      !out_dir.empty(),    // dump_passes
      false,               // dump_passes_proto
      false,               // dump_code
      out_dir / "passes",  // dbg_dir
  };
  const auto& cfgs = targets::GetConfigs();
  const auto& cfg = cfgs.configs().at(target);
  const auto& stage = cfg.stages().at("default");
  codegen::CompilerState state(stripe);
  state.const_bufs = const_bufs;
  codegen::Optimize(&state, stage.passes(), options);
  targets::cpu::Config config;
  if (!env::Get("PLAIDML_CPU_PROFILE").empty()) {
    config.profile_block_execution = true;
    source_ = CloneBlock(*stripe->entry);
  }
  executable_->compile(*(source_ ? source_ : stripe->entry), config);
}

CpuProgram::~CpuProgram() {}

boost::future<void> CpuProgram::Run(  //
    const context::Context& ctx,      //
    std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
    std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) {
  std::map<std::string, void*> buffers;
  // map in the input buffers, preserving contents
  for (auto& kvp : inputs) {
    IVLOG(2, "Input: " << kvp.first);
    auto view = kvp.second->MapCurrent(ctx).get();
    buffers.emplace(kvp.first, view->data());
  }
  // map in output buffers, discarding contents
  for (auto& kvp : outputs) {
    IVLOG(2, "Output: " << kvp.first);
    // don't overwrite the buffer if it's already been mapped in
    if (buffers.find(kvp.first) == buffers.end()) {
      auto view = kvp.second->MapDiscard(ctx);
      buffers.emplace(kvp.first, view->data());
    }
  }
  executable_->run(buffers);
  std::string profile_var = env::Get("PLAIDML_CPU_PROFILE");
  if (!profile_var.empty()) {
    // copy profile measurements into the saved stripe block
    executable_->set_perf_attrs(source_.get());
    // generate a unique file name for this run
    static unsigned run_counter;
    std::string suffix = "00000" + std::to_string(run_counter++);
    suffix = suffix.substr(suffix.size() - 6);
    auto path = boost::filesystem::path(profile_var + "." + suffix);
    // dump annotated stripe block contents to disk
    std::ofstream fout(path.string());
    fout << *source_ << std::endl;
  }
  return boost::make_ready_future();
}

void CpuProgram::Release() {}

std::size_t CpuProgram::MaxAvailableMemory() {
  throw std::runtime_error("CpuProgram::MaxAvailableMemory is unimplemented");
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
