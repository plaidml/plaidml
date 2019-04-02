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

KernelList GenerateProgram(const Program& prog,          //
                           const ShapeMap& inputs,       //
                           const ShapeMap& outputs,      //
                           const std::string& cfg_name,  //
                           const std::string& out_dir) {
  IVLOG(1, inputs);
  IVLOG(1, outputs);
  IVLOG(1, to_string(prog));
  ShapeMap all;
  auto stripe = GenerateStripe(prog, inputs, outputs, &all);
  codegen::OptimizeOptions options = {
      !out_dir.empty(),     // dump_passes
      false,                // dump_code
      out_dir + "/passes",  // dbg_dir
  };
  IVLOG(1, *stripe);
  auto cfg = Configs::Resolve(cfg_name);
  codegen::Optimize(stripe.get(), cfg.passes(), options);
  IVLOG(1, *stripe);
  codegen::SemtreeEmitter emit(codegen::AliasMap{}, 256);
  emit.Visit(*stripe);
  // lang::Simplify(emit.kernels_.kernels);
  for (const auto ki : emit.kernels_.kernels) {
    sem::Print p(*ki.kfunc);
    IVLOG(1, p.str());
    IVLOG(1, "gids = " << ki.gwork);
    IVLOG(1, "lids = " << ki.lwork);
  }
  IVLOG(1, "RETURNING THOSE KERNELS");
  AliasMap init_map;
  AliasMap prog_map(init_map, stripe.get());
  AliasMap main_map(prog_map, stripe->SubBlock(0).get());
  for (const auto& ref : stripe->SubBlock(0)->refs) {
    if (ref.dir != stripe::RefDir::None) {
      emit.kernels_.types[ref.from] = ref.interior_shape;
    } else {
      emit.kernels_.types["local_" + ref.into] = ref.interior_shape;
    }
  }
  for (auto& ki : emit.kernels_.kernels) {
    for (auto& name : ki.inputs) {
      const auto& ai = main_map.at(name);
      name = ai.base_block == stripe.get() ? ai.base_ref->into : ("local_" + name);
    }
    for (auto& name : ki.outputs) {
      const auto& ai = main_map.at(name);
      name = ai.base_block == stripe.get() ? ai.base_ref->into : ("local_" + name);
    }
  }
  return emit.kernels_;
}

}  // End namespace codegen
}  // End namespace tile
}  // End namespace vertexai
