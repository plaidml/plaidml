#include "tile/lang/generate.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "base/util/logging.h"
#include "tile/lang/compile.h"
#include "tile/lang/flat.h"
#include "tile/lang/fpconv.h"
#include "tile/lang/gen_contract.h"
#include "tile/lang/gen_elemwise.h"
#include "tile/lang/gen_special.h"
#include "tile/lang/gid.h"
#include "tile/lang/ops.h"
#include "tile/lang/parser.h"
#include "tile/lang/tile_opt.h"
#include "tile/lang/type.h"
#include "tile/lang/usedef.h"

namespace vertexai {
namespace tile {
namespace lang {

static bool NeedsZero(const FlatContraction& flat, const TensorShape& ts) {
  std::vector<std::pair<size_t, size_t>> out_pattern;
  if (flat.access[0].offset != 0) {
    return true;
  }
  for (size_t i = 0; i < flat.names.size(); i++) {
    if (flat.access[0].strides[i] == 0) {
      continue;
    }
    if (flat.access[0].strides[i] < 0) {
      return true;
    }  // Don't try to be fancy, fallback
    out_pattern.emplace_back(flat.access[0].strides[i], flat.ranges[i]);
  }
  for (const FlatConstraint& fc : flat.constraints) {
    bool output_only = true;
    for (size_t i = 0; i < flat.names.size(); i++) {
      if (fc.lhs[i] != 0 && flat.access[0].strides[i] == 0) {
        output_only = false;
        break;
      }
    }
    if (output_only) {
      return true;
    }
  }
  std::sort(out_pattern.begin(), out_pattern.end());
  size_t curskip = 1;
  for (const auto& p : out_pattern) {
    if (curskip != p.first) {
      return true;
    }
    curskip *= p.second;
  }
  return curskip != flat.access[0].global_index_limit;
}

static KernelInfo GenerateContractionKernel(const std::string& kname, const HardwareSettings& settings,
                                            const Contraction& c, const FlatContraction& flat,
                                            const std::vector<uint64_t>& tile, const std::vector<std::string>& inputs,
                                            const std::vector<std::string>& outputs, const Bindings& vars) {
  KernelInfo ki = GenContract(kname, settings, flat, tile, vars, inputs);
  ki.outputs = outputs;
  ki.key = flat.KeyString();
  ki.settings = settings;
  ki.tile_size = tile;
  for (const auto& input : inputs) {
    if (vars.at(input).tag == Binding::TENSOR) {
      ki.inputs.emplace_back(input);
    }
  }
  PerfStats perf = ComputeTileStats(settings, flat, tile);
  ki.tot_bytes = perf.work_groups * ((perf.inner_loops * perf.mem_read) + perf.mem_write);
  ki.tot_flops = perf.true_ops;
  if (VLOG_IS_ON(1)) {
    std::string tsize = "";
    for (size_t size : tile) {
      tsize += std::to_string(size) + ", ";
    }
    VLOG(1) << "Contraction " << kname << ":\n"
            << to_string(c) << "\n"
            << to_string(flat) << "\n"
            << tsize << "\n"
            << "tot_flops = " << ki.tot_flops << ", tot_bytes = " << ki.tot_bytes << "\n\n";
  }
  auto pb = ki.info.mutable_contraction();
  pb->set_op(to_string(c));
  for (std::size_t idx = 0; idx < flat.names.size(); ++idx) {
    auto access = pb->add_accesses();
    access->set_name(flat.names[idx]);
    access->set_range(flat.ranges[idx]);
    for (auto a : flat.access) {
      access->add_strides(a.strides[idx]);
    }
  }
  for (auto a : flat.access) {
    pb->add_off(a.offset);
    pb->add_vec(a.vector);
  }
  for (auto c : flat.constraints) {
    auto constraint = pb->add_constraints();
    for (auto lhs : c.lhs) {
      constraint->add_lhs(lhs);
    }
    constraint->set_rhs(c.rhs);
  }

  return ki;
}

static void ContractionWrap(KernelList& r, const Contraction& c, const ShapeMap& shapes,  // NOLINT(runtime/references)
                            const std::string& kname, const HardwareSettings& settings, const Bindings& vars,
                            size_t tile_trials) {
  if (c.specs.size() != 2 && c.specs.size() != 3 && c.specs.size() != 4) {
    throw std::runtime_error("Currently, we only support 1, 2, and 3 element Contractions");
  }
  std::vector<TensorShape> tshapes;
  std::vector<std::string> outputs;
  std::vector<std::string> inputs;
  bool first = true;
  for (const TensorSpec& spec : c.specs) {
    auto it = shapes.find(spec.id);
    if (it == shapes.end()) {
      IVLOG(1, "About to barf: " << shapes);
      throw std::runtime_error(printstring("Unable to find tensor shape for id %s, ug", spec.id.c_str()));
    }
    tshapes.push_back(it->second);
    if (first) {
      outputs.push_back(it->first);
    } else {
      inputs.push_back(it->first);
    }
    first = false;
  }
  FlatContraction flat = Compile(c, tshapes);
  if (NeedsZero(flat, tshapes[0])) {
    r.kernels.push_back(GenZero(tshapes[0], outputs[0], "zero_" + kname));
  }
  // Do memory based tile optimization
  if (settings.vec_size > 1) {
    flat = Vectorize(flat, settings.vec_size);
  }
  auto by_score = TileOptimize(settings, flat, tile_trials == 1);

  KernelInfo primary;
  size_t trial_count = 0;
  for (auto it = by_score.rbegin(); it != by_score.rend() && trial_count < tile_trials; it++, trial_count++) {
    auto tile = it->second;
    KernelInfo ki = GenerateContractionKernel(kname, settings, c, flat, tile, inputs, outputs, vars);
    if (trial_count == 0) {
      primary = ki;
    } else {
      primary.candidates.push_back(ki);
    }
  }
  r.kernels.push_back(primary);
}

static KernelList Compile(const Program& orig_prog, const ShapeMap& inputs, const ShapeMap& outputs,
                          const HardwareSettings& settings, const std::string& kid, size_t tile_trials) {
  IVLOG(2, "Compile");
  KernelList r;
  Program prog = orig_prog;
  Bindings vars = BindProgram(&prog, inputs, outputs);
  // Move to a shapemap for compatibility
  ShapeMap types;
  for (const auto& kvp : vars) {
    types.emplace(kvp.first, kvp.second.shape);
  }

  // Make a convolution kernel for each convolution, and a function kernel for
  // each group of connected functions.

  // First, compute use/defs for later use
  UseDef ud(prog);

  // Now, go over all of the program operations
  std::set<size_t> computed;
  size_t knum = 0;
  for (size_t i = 0; i < prog.ops.size(); i++) {
    const Op& op = prog.ops[i];
    if (op.tag == Op::CONTRACTION) {
      // If it's a contraction, do the easy thing
      IVLOG(3, "Running contraction " << op << " types = " << types);
      ContractionWrap(r, op.c, types, printstring("%s_%zu", kid.c_str(), knum++), settings, vars, tile_trials);
      continue;
    }
    // Ignore constants
    if (op.tag == Op::CONSTANT) {
      continue;
    }
    // Ignore already computed programs
    if (computed.count(i)) {
      continue;
    }
    // Special handling for special functions
    if (op.f.is_special()) {
      Op dop = op;
      if (op.f.fn == "prng_state" || op.f.fn == "prng_value") {
        throw std::runtime_error("prng functions must come in threes");
      }
      if (op.f.fn == "prng_step") {
        std::string tup = op.output;
        std::string sout;
        std::string vout;
        size_t sout_pos = 0;
        // Find the other parts
        for (size_t j = i + 1; j < prog.ops.size(); j++) {
          const Op& nop = prog.ops[j];
          if (nop.f.fn == "prng_state" && nop.inputs.size() == 1 && nop.inputs[0] == tup) {
            sout = nop.output;
            sout_pos = j;
            computed.emplace(j);
          } else if (nop.f.fn == "prng_value" && nop.inputs.size() == 1 && nop.inputs[0] == tup) {
            vout = nop.output;
            computed.emplace(j);
          }
        }
        if (vout == "" && sout == "") {
          continue;  // Skip the whole thing
        }
        if (vout == "") {
          // Convert state output to identity
          Op& xop = prog.ops[sout_pos];
          xop.f.fn = "ident";
          xop.inputs[0] = op.inputs[0];
          computed.erase(sout_pos);
          continue;
        }
        if (sout == "") {
          throw std::runtime_error("prng_step function missing its compainions");
        }
        dop.f.params.push_back(sout);
        dop.f.params.push_back(vout);
      }
      GenSpecial(r, dop, vars, printstring("%s_%zu", kid.c_str(), knum++), settings);
      continue;
    }
    // Otherwise, find the connected components
    std::set<size_t> comps = ud.ConnectedComponents(prog, i, computed);
    IVLOG(3, "CC = " << comps);
    // Add to list of computed parts
    computed.insert(comps.begin(), comps.end());
    // Compile function (TODO: do something with the output)
    KernelInfo ki =
        GenFunction(prog, outputs, types, vars, comps, printstring("%s_%zu", kid.c_str(), knum++), ud, settings);
    r.kernels.push_back(std::move(ki));
  }

  // Copy only the relevant typing info across
  for (const KernelInfo& ki : r.kernels) {
    for (const std::string& s : ki.inputs) {
      r.types[s] = types[s];
    }
    for (const std::string& s : ki.outputs) {
      r.types[s] = types[s];
    }
  }
  return r;
}

KernelList GenerateProgram(const Program& prog, const ShapeMap& inputs, const ShapeMap& outputs,
                           const HardwareSettings& settings, const std::string& id, size_t tile_trials) {
  // The caller can pass whatever it likes as the program ID, but for OpenCL, we require a valid C identifier.  We do
  // this by prefixing the supplied identifier with "kernel_" and translating all non-alnum characters to '_'.
  IVLOG(1, "Doing a compilation of:\n" << to_string(prog) << "\n");
  std::string kid = "kernel_";
  kid.reserve(kid.size() + id.size());
  for (char c : id) {
    if (!std::isalnum(c)) {
      c = '_';
    }
    kid.push_back(c);
  }

  // Do the primary compilations
  KernelList r;
  r = Compile(prog, inputs, outputs, settings, kid, tile_trials);

  return r;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
