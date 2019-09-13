#include "tile/ocl_exec/stripe_gen.h"

#include <stdio.h>

#include "base/util/file.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/driver.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/semprinter.h"
#include "tile/lang/simplifier.h"
#include "tile/ocl_exec/emitsem.h"
#include "tile/targets/targets.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace lang;  // NOLINT

lang::KernelList GenerateProgram(                    //
    const std::shared_ptr<stripe::Program>& stripe,  //
    const std::string& cfg_name,                     //
    const std::string& out_dir,                      //
    ConstBufferManager* const_bufs) {
  codegen::OptimizeOptions options;
  options.dump_passes = !out_dir.empty();
  options.dump_passes_proto = !out_dir.empty();
  options.dbg_dir = out_dir + "/passes";
  IVLOG(2, "Write passes to: " << options.dbg_dir);
  IVLOG(2, *stripe->entry);
  const auto& cfgs = targets::GetConfigs();
  const auto& cfg = cfgs.configs().at(cfg_name);
  const auto& stage = cfg.stages().at("default");
  codegen::CompilerState state(stripe);
  state.const_bufs = const_bufs;
  codegen::Optimize(&state, stage.passes(), options);
  IVLOG(2, *stripe->entry);
  codegen::SemtreeEmitter emit(codegen::AliasMap{}, 256);
  emit.Visit(*stripe->entry);
  // lang::Simplify(emit.kernels_.kernels);
  if (VLOG_IS_ON(2)) {
    for (const auto ki : emit.kernels_.kernels) {
      sem::Print p(*ki.kfunc);
      IVLOG(2, p.str());
      IVLOG(2, "gids = " << ki.gwork);
      IVLOG(2, "lids = " << ki.lwork);
    }
  }
  auto main = stripe->entry->SubBlock(0);
  AliasMap init_map;
  AliasMap prog_map(init_map, stripe->entry.get());
  AliasMap main_map(prog_map, main.get());
  for (const auto& ref : main->refs) {
    if (ref.dir != stripe::RefDir::None) {
      emit.kernels_.types[ref.from] = ref.interior_shape;
    } else {
      emit.kernels_.types["local_" + ref.into()] = ref.interior_shape;
    }
  }
  for (auto& ki : emit.kernels_.kernels) {
    for (auto& name : ki.inputs) {
      const auto& ai = main_map.at(name);
      name = ai.base_block == stripe->entry.get() ? ai.base_ref->into() : ("local_" + name);
    }
    for (auto& name : ki.outputs) {
      const auto& ai = main_map.at(name);
      name = ai.base_block == stripe->entry.get() ? ai.base_ref->into() : ("local_" + name);
    }
  }
  return emit.kernels_;
}

KernelList GenerateProgram(       //
    const RunInfo& runinfo,       //
    const std::string& cfg_name,  //
    const std::string& out_dir,   //
    ConstBufferManager* const_bufs) {
  IVLOG(2, runinfo.input_shapes);
  IVLOG(2, runinfo.output_shapes);
  IVLOG(2, to_string(runinfo.program));
  auto stripe = GenerateStripe(runinfo);
  return GenerateProgram(stripe, cfg_name, out_dir, const_bufs);
}

}  // End namespace codegen
}  // End namespace tile
}  // End namespace vertexai
