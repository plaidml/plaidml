#include "tile/ocl_exec/stripe_gen.h"

#include <stdio.h>

#include "base/config/config.h"
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

KernelList GenerateProgram(const Program& prog, const ShapeMap& inputs, const ShapeMap& outputs,
                           const std::string& cfg_file, const std::string& out_dir) {
  IVLOG(1, inputs);
  IVLOG(1, outputs);
  IVLOG(1, to_string(prog));
  ShapeMap all;
  auto stripe = GenerateStripe(prog, inputs, outputs, &all);
  auto cfg = ParseConfig<codegen::proto::Config>(ReadFile(cfg_file));
  codegen::OptimizeOptions options = {
      !out_dir.empty(),     // dump_passes
      false,                // dump_code
      out_dir + "/passes",  // dbg_dir
  };
  IVLOG(1, *stripe);
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
  emit.kernels_.types = all;
  return emit.kernels_;
}

}  // End namespace codegen
}  // End namespace tile
}  // End namespace vertexai
