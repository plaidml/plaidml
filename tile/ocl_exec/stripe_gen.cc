#include "tile/ocl_exec/stripe_gen.h"

#include <stdio.h>

#include "base/util/file.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/driver.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/semprinter.h"
#include "tile/lang/simplifier.h"
#include "tile/ocl_exec/emitsem.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace lang;  // NOLINT

KernelList GenerateProgram(const RunInfo& runinfo,       //
                           const std::string& cfg_name,  //
                           const std::string& out_dir) {
  IVLOG(1, runinfo.input_shapes);
  IVLOG(1, runinfo.output_shapes);
  IVLOG(1, to_string(runinfo.program));
  auto stripe = GenerateStripe(runinfo);

  codegen::OptimizeOptions options;
  options.dump_passes = !out_dir.empty();
  options.dbg_dir = out_dir + "/passes";
  IVLOG(1, *stripe->entry);
  auto cfg = Configs::Resolve(cfg_name);
  codegen::Optimize(stripe->entry.get(), cfg.passes(), options);
  IVLOG(1, *stripe->entry);
  codegen::SemtreeEmitter emit(codegen::AliasMap{}, 256);
  emit.Visit(*stripe->entry);
  // lang::Simplify(emit.kernels_.kernels);
  for (const auto ki : emit.kernels_.kernels) {
    sem::Print p(*ki.kfunc);
    IVLOG(1, p.str());
    IVLOG(1, "gids = " << ki.gwork);
    IVLOG(1, "lids = " << ki.lwork);
  }
  IVLOG(1, "RETURNING THOSE KERNELS");
  AliasMap init_map;
  AliasMap prog_map(init_map, stripe->entry.get());
  AliasMap main_map(prog_map, stripe->entry->SubBlock(0).get());
  for (const auto& ref : stripe->entry->SubBlock(0)->refs) {
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

}  // End namespace codegen
}  // End namespace tile
}  // End namespace vertexai
