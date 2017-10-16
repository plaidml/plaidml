#include "tile/lang/generate.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "base/util/logging.h"
#include "tile/lang/compile.h"
#include "tile/lang/flat.h"
#include "tile/lang/fpconv.h"
#include "tile/lang/gen_contract.h"
#include "tile/lang/gen_special.h"
#include "tile/lang/gen_zero.h"
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
                                            const Contraction* c, const FlatContraction& flat,
                                            const std::vector<uint64_t>& tile, const std::vector<std::string>& inputs,
                                            const Bindings& vars) {
  KernelInfo ki = GenContract(kname, settings, flat, tile, vars, inputs);
  ki.outputs = flat.kernel_outputs;
  ki.key = flat.KeyString();
  ki.settings = settings;
  ki.tile_size = tile;
  for (const auto& input : inputs) {
    if (vars.at(input).tag == Binding::TENSOR) {
      ki.inputs.emplace_back(input);
    }
  }
  for (const auto& kvp : flat.post_op_inputs) {
    ki.inputs.emplace_back(kvp.first);
  }
  PerfStats perf = ComputeTileStats(settings, flat, tile, vars);
  ki.tot_bytes = perf.work_groups * ((perf.inner_loops * perf.mem_read) + perf.mem_write);
  ki.tot_flops = perf.true_ops;
  if (VLOG_IS_ON(1)) {
    std::string tsize = "";
    for (size_t size : tile) {
      tsize += std::to_string(size) + ", ";
    }
    VLOG(1) << "Contraction " << kname << ":\n"
            << (c ? to_string(*c) : "<empty>") << "\n"
            << to_string(flat) << "\n"
            << tsize << "\n";
    if (flat.post_ops.size()) {
      VLOG(1) << "Output operations:";
      for (const auto& op : flat.post_ops) {
        VLOG(1) << "  " << op;
      }
    }
    VLOG(1) << "tot_flops = " << ki.tot_flops << ", tot_bytes = " << ki.tot_bytes << "\n\n";
  }
  if (c) {
    auto pb = ki.info.mutable_contraction();
    pb->set_op(to_string(*c));
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
    for (auto cons : flat.constraints) {
      auto constraint = pb->add_constraints();
      for (auto lhs : cons.lhs) {
        constraint->add_lhs(lhs);
      }
      constraint->set_rhs(cons.rhs);
    }
  }

  return ki;
}

static std::vector<TensorShape> MakeTShapes(const Contraction& c, const Bindings& vars) {
  std::vector<TensorShape> tshapes;
  for (const TensorSpec& spec : c.specs) {
    auto it = vars.find(spec.id);
    if (it == vars.end()) {
      IVLOG(1, "About to barf: " << vars);
      throw std::runtime_error(printstring("Unable to find tensor shape for id %s, ug", spec.id.c_str()));
    }
    tshapes.push_back(it->second.shape);
  }
  return tshapes;
}

// Simplify a flat contraction by combining indexes if possible
static bool SimplifyFlat(FlatContraction* flat) {
  // Skip if we have any constraints, cuz it's tricky
  if (flat->constraints.size() > 0) {
    return false;
  }
  // This algorithm is n^3 at worst (n calls to flatten, each doing n^2 work)
  // Hopefully n is pretty small.
  size_t sz = flat->ranges.size();
  for (size_t i = 0; i < sz; i++) {
    size_t i_stride = flat->access[0].strides[i];
    if (i_stride == 0) {
      continue;
    }
    for (size_t j = 0; j < sz; j++) {
      size_t j_stride = flat->access[0].strides[j];
      if (j_stride == 0) {
        continue;
      }
      if (i_stride != flat->ranges[j] * j_stride) {
        continue;
      }
      auto is_safe = [&](const FlatTensorAccess& a) -> bool {
        bool perfect_match = (a.strides[i] == i_stride && a.strides[j] == j_stride);
        bool both_zeros = (a.strides[i] == 0 && a.strides[j] == 0);
        return perfect_match || perfect_match;
      };
      bool all_good = true;
      for (size_t k = 1; k < flat->access.size(); k++) {
        if (!is_safe(flat->access[k])) {
          all_good = false;
          break;
        }
      }
      for (const auto& kvp : flat->post_op_inputs) {
        if (!is_safe(kvp.second)) {
          all_good = false;
          break;
        }
      }
      if (!all_good) {
        continue;
      }
      IVLOG(3, "SimplifyFlat: Combining " << flat->names[i] << " and " << flat->names[j]);
      IVLOG(3, "Pre=\n" << to_string(*flat));
      // Found a valid indexes to combine!
      flat->names[j] = flat->names[i] + "_" + flat->names[j];
      flat->names.erase(flat->names.begin() + i);
      flat->ranges[j] *= flat->ranges[i];
      flat->ranges.erase(flat->ranges.begin() + i);
      auto fixup = [&](FlatTensorAccess& a) { a.strides.erase(a.strides.begin() + i); };
      for (size_t k = 0; k < flat->access.size(); k++) {
        fixup(flat->access[k]);
      }
      for (auto& kvp : flat->post_op_inputs) {
        fixup(kvp.second);
      }
      IVLOG(3, "Out=\n" << to_string(*flat));
      // We bail and let the outer caller rerun the main loop
      // This is mostly because the indexes we are iterating over changed
      // and thinking is hard.
      return true;
    }
  }
  return false;
}

static void ContractionWrap(KernelList& r, const Contraction* c, FlatContraction flat,  // NOLINT(runtime/references)
                            const std::string& kname, const HardwareSettings& settings, const Bindings& vars,
                            size_t tile_trials) {
  std::vector<std::string> inputs;
  if (c) {
    if (c->specs.size() != 2 && c->specs.size() != 3 && c->specs.size() != 4) {
      throw std::runtime_error("Currently, we only support 1, 2, and 3 element Contractions");
    }
    bool first = true;
    for (const TensorSpec& spec : c->specs) {
      auto it = vars.find(spec.id);
      if (it == vars.end()) {
        IVLOG(1, "About to barf: " << vars);
        throw std::runtime_error(printstring("Unable to find tensor shape for id %s, ug", spec.id.c_str()));
      }
      if (!first) {
        inputs.push_back(it->first);
      }
      first = false;
    }
  }
  // Flatten out needless dimensions
  while (SimplifyFlat(&flat)) {
  }
  // Do memory based tile optimization
  if (settings.vec_size > 1) {
    flat = Vectorize(flat, settings.vec_size);
  }
  IVLOG(4, "Optimizing " << kname);
  auto by_score = TileOptimize(settings, flat, tile_trials == 1, vars);

  KernelInfo primary;
  size_t trial_count = 0;
  for (auto it = by_score.rbegin(); it != by_score.rend() && trial_count < tile_trials; it++, trial_count++) {
    auto tile = it->second;
    KernelInfo ki = GenerateContractionKernel(kname, settings, c, flat, tile, inputs, vars);
    if (trial_count == 0) {
      primary = ki;
    } else {
      primary.candidates.push_back(ki);
    }
  }
  r.kernels.push_back(primary);
}

static bool DifferentDims(const Binding& a, const Binding& b) {
  if (a.tag != Binding::TENSOR || b.tag != Binding::TENSOR) {
    return true;
  }
  return a.shape.dims != b.shape.dims;
}

static void DoUnification(FlatContraction* flat, std::set<std::size_t>* computed, const Program& prog,
                          std::size_t opidx, const UseDef& ud, const Bindings& vars, const ShapeMap& inputs,
                          const ShapeMap& outputs, const std::vector<Polynomial>& out_poly) {
  // Unify the contraction with downstream elementwise operations.
  //
  // Here's the idea: during the contraction's output phase, we
  // have some set of outputs available, starting with the
  // actual output of the contraction.  So we scan the uses of
  // those outputs: any downstream elementwise operation that's
  // only dependent on the outputs we have so far, program
  // inputs, or constants, can be unified into the current
  // contraction.  Elementwise operations that are added to a
  // contraction add their own outputs to the set of outputs
  // available, thus allowing further elementwise operations to
  // be added.

  const Op& op = prog.ops[opidx];

  // Additional inputs required for the unified kernel.
  std::set<std::string> post_contraction_inputs;

  // The set of outputs that are known to be available within the kernel.
  std::set<std::string> available_outputs;

  // The set of elementwise operations that have been unified with the kernel.
  std::set<std::size_t> unified_opidxs;

  // The set of operations that have been unified with the kernel
  // (starting with the initiating contraction or elementwise
  // operation) whose outputs need to checked for downstream
  // elementwise operations that might be unified.
  //
  // We always check these lowest-numbered-op-first, since the
  // operation list is in single-assignment form, and consumers
  // are always after producers in the list.
  std::set<std::size_t> ops_to_check;

  // Initialize the set of outputs available.
  available_outputs.insert(op.output);
  unified_opidxs.insert(opidx);

  if (!flat->generate_contraction) {
    // If there's no actual contraction, the initiating operation has
    // already been added to the FlatContraction's post-ops list.
    // Initialize the post_contraction_inputs to take this into
    // account -- the operation's inputs are required to come from
    // kernel parameters.
    for (const auto& input : op.inputs) {
      if (vars.at(input).tag == Binding::TENSOR) {
        post_contraction_inputs.emplace(input);
      }
    }
  }

  // Add the initial operation's output's consumers as the ops to check.
  {
    auto use_it = ud.uses().find(op.output);
    if (use_it != ud.uses().end()) {
      ops_to_check.insert(use_it->second.begin(), use_it->second.end());
    }
  }

  IVLOG(3, "In unification, out polys = " << out_poly);
  IVLOG(4, "Looking for ops to unify with op " << op);

  while (ops_to_check.size()) {
    // Pop an operation to be checked for possible unification.
    auto check_it = ops_to_check.begin();
    auto check_opidx = *check_it;
    ops_to_check.erase(check_it);

    auto& check_op = prog.ops[check_opidx];
    IVLOG(4, "  Checking op " << check_op);

    if (check_opidx != opidx) {  // The initial operation is automatically unified.
      if (check_op.tag != Op::FUNCTION || check_op.f.is_special()) {
        IVLOG(4, "  Consumer tag=" << check_op.tag << " inputs.size=" << check_op.inputs.size()
                                   << " is_special=" << check_op.f.is_special() << "; skipping unification");
        continue;
      }
      if (DifferentDims(vars.at(op.output), vars.at(check_op.output))) {
        IVLOG(4, "  Var " << op.output << " differs in dimensions from " << check_op.output
                          << "; skipping unification");
        continue;
      }

      bool all_inputs_available = true;
      std::vector<std::string> check_op_added_inputs;
      for (const auto& input : check_op.inputs) {
        if (vars.at(input).tag != Binding::TENSOR) {
          continue;
        }
        if (available_outputs.count(input)) {
          // We've merged this input's creator into this op.
          continue;
        }
        if (inputs.count(input)) {
          // This is a program input.
          check_op_added_inputs.push_back(input);
          continue;
        }
        // Tensor inputs that aren't in the program inputs should be in the usedef map.
        assert(ud.op_defs().count(input));

        // If the input was generated by an earlier operation, we
        // can add it as an input to the current kernel, enabling
        // merging of the operation we're checking.
        //
        // It's not clear that this is always a good idea,
        // since it prevents the earlier operation and the
        // current operation from running in parallel.
        size_t src_num = ud.op_defs().at(input);
        if (src_num <= opidx || computed->count(src_num)) {
          check_op_added_inputs.push_back(input);
          continue;
        }

        all_inputs_available = false;
        break;
      }
      if (!all_inputs_available) {
        IVLOG(4, "  Op " << check_op << " cannot be computed in this contraction; skipping unification");
        continue;
      }

      IVLOG(4, "  Scheduling unification of op " << check_op);
      // Looks like this elementwise op can be unified with the current contraction.
      flat->post_ops.emplace_back(check_op);
      unified_opidxs.insert(check_opidx);
      available_outputs.insert(check_op.output);
      post_contraction_inputs.insert(std::make_move_iterator(check_op_added_inputs.begin()),
                                     std::make_move_iterator(check_op_added_inputs.end()));
    }

    // Add the uses of the op's outputs for consideration.
    auto use_it = ud.uses().find(check_op.output);
    if (use_it != ud.uses().end()) {
      ops_to_check.insert(use_it->second.begin(), use_it->second.end());
    }
  }

  // For all available outputs: if the usedefs or program
  // outputs require it, add it to the kernel outputs.
  for (auto unified_opidx : unified_opidxs) {
    auto& unified_op = prog.ops[unified_opidx];
    bool needed_as_output = false;
    if (outputs.count(unified_op.output)) {
      // It's a program output; we need to write it.
      needed_as_output = true;
    } else {
      auto use_it = ud.uses().find(unified_op.output);
      if (use_it != ud.uses().end()) {
        for (auto use_opidx : use_it->second) {
          if (!unified_opidxs.count(use_opidx)) {
            needed_as_output = true;
            break;
          }
        }
      }
    }

    if (needed_as_output) {
      flat->kernel_outputs.push_back(unified_op.output);
    }
  }

  // Copy over post contraction inputs and compute strides
  computed->insert(unified_opidxs.begin(), unified_opidxs.end());
  const TensorShape& out_shape = vars.at(flat->output).shape;
  for (const auto& name : post_contraction_inputs) {
    const TensorShape& shape = vars.at(name).shape;
    FlatTensorAccess a;
    a.global_index_limit = shape.buffer_size();
    Polynomial p;
    size_t off = out_poly.size() - shape.dims.size();
    for (size_t i = 0; i < shape.dims.size(); i++, off++) {
      // We add things if they are not broadcast, we treat 1, 1 as non broadcast in this case
      if (shape.dims[i].size != 1 || out_shape.dims[off].size == 1) {
        p += out_poly[off] * shape.dims[i].stride;
      }
    }
    for (const auto& idx : flat->names) {
      a.strides.push_back(static_cast<int64_t>(Floor(p[idx])));
    }
    IVLOG(3, "For shape: " << shape << " poly = " << p << " strides = " << a.strides);
    flat->post_op_inputs.emplace(name, a);
  }
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

  // First, compute use/defs for later use
  UseDef ud(prog);

  // Remember the set of operations that have already been covered by kernels
  // (necessary since a given kernel may encompass multiple ops).
  std::set<size_t> computed;

  // Now, go over all of the program operations; make a convolution kernel for each convolution, and a function kernel
  // for each group of connected functions.
  size_t knum = 0;
  auto next_kname = [&knum, kid] { return printstring("%s_%zu", kid.c_str(), knum++); };
  time_t last_update = time(nullptr);
  for (size_t i = 0; i < prog.ops.size(); i++) {
    if (time(nullptr) - last_update >= 2) {
      LOG(INFO) << "Analysing Ops: " << i << " of " << prog.ops.size() << " operations complete";
      last_update = time(nullptr);
    }
    const Op& op = prog.ops[i];

    if (op.tag == Op::CONTRACTION) {
      IVLOG(3, "Running contraction " << op << " vars = " << vars);
      std::vector<TensorShape> tshapes = MakeTShapes(op.c, vars);
      std::vector<Polynomial> out_poly;
      FlatContraction flat = Compile(op.c, tshapes, &out_poly);
      flat.output = op.output;

      auto kname = next_kname();
      if (NeedsZero(flat, tshapes[0])) {
        // N.B. We currently don't unify kernels with subsequent
        // operations unless they cover the entire output space.
        r.kernels.push_back(GenZero(tshapes[0], op.output, "zero_" + kname));
        flat.kernel_outputs.push_back(op.output);
      } else {
        DoUnification(&flat, &computed, prog, i, ud, vars, inputs, outputs, out_poly);
      }
      ContractionWrap(r, &op.c, std::move(flat), kname, settings, vars, tile_trials);
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
      GenSpecial(r, dop, vars, next_kname(), settings);
      continue;
    }

    // Otherwise, it's an elementwise operation that hasn't been
    // unified with an earlier contraction.  Initialize a
    // FlatContraction object to represent the computation to the rest
    // of the tile shaping logic; we'll omit generating the
    // contraction itself later.

    FlatContraction flat;
    std::vector<Polynomial> out_poly;
    {
      // The initial elementwise operation's output is used to
      // determine the shape of the overall kernel -- which is
      // reasonable, because every subsequent elementwise operation is
      // required to have an output that's same shape as that initial
      // operation.
      flat.generate_contraction = false;

      const auto& access_op = prog.ops[i];
      flat.post_ops.emplace_back(access_op);

      flat.output = access_op.output;
      const TensorShape& shape = vars.at(access_op.output).shape;
      for (std::size_t idx = 0; idx < shape.dims.size(); ++idx) {
        std::string idx_name = std::string("i") + std::to_string(idx + 1);
        flat.names.push_back(idx_name);
        out_poly.push_back(Polynomial(idx_name));
        flat.ranges.push_back(shape.dims[idx].size);
      }

      FlatTensorAccess access;
      access.type = shape.type;
      access.vector = 1;
      access.offset = 0;
      access.global_index_limit = shape.buffer_size();
      for (const auto& dim : shape.dims) {
        access.strides.emplace_back(dim.stride);
      }
      flat.access.emplace_back(std::move(access));
    }

    DoUnification(&flat, &computed, prog, i, ud, vars, inputs, outputs, out_poly);

    ContractionWrap(r, nullptr, std::move(flat), next_kname(), settings, vars, tile_trials);
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
