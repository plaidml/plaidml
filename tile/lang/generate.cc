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
#include "tile/lang/gen_trivial.h"
#include "tile/lang/ops.h"
#include "tile/lang/parser.h"
#include "tile/lang/simplifier.h"
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
                                            const Contraction* c, const FlatContraction& flat, const TileOption& option,
                                            const std::vector<std::string>& inputs, const Bindings& vars,
                                            const VarRewrites& var_rewrites) {
  proto::PerfStats perf = ComputeTileStats(settings, flat, option.shape);
  KernelInfo ki = GenContract(kname, settings, flat, option.shape, vars, inputs, perf);
  ki.outputs = flat.kernel_outputs;
  ki.key = flat.TileKeyString();
  ki.settings = settings;
  ki.flat = flat;
  ki.tile = option;
  for (const auto& input : inputs) {
    if (vars.at(input).tag == Binding::TENSOR) {
      ki.inputs.emplace_back(var_rewrites.Lookup(input));
    }
  }
  for (const auto& op_input : flat.post_op_inputs) {
    ki.inputs.emplace_back(var_rewrites.Lookup(op_input.name));
  }
  ki.tot_bytes = perf.work_groups() * ((perf.inner_loops() * perf.mem_read()) + perf.mem_write());
  ki.tot_flops = perf.true_ops();
  *(ki.info.mutable_perf_stats()) = perf;
  if (VLOG_IS_ON(1)) {
    std::string tsize = "";
    for (size_t size : option.shape) {
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

  proto::ContractionInfo* pb;
  if (c) {
    pb = ki.info.mutable_contraction();
    pb->add_ops(to_string(*c));
  } else {
    pb = ki.info.mutable_element();
  }
  for (const auto& op : flat.post_ops) {
    pb->add_ops(to_string(op));
  }
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
  ki.info.set_flops(ki.tot_flops);
  ki.info.set_bytes(ki.tot_bytes);

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
  // Skip any case where we use the index builtin
  for (const auto& op : flat->post_ops) {
    if (op.f.fn == "index") {
      return false;
    }
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
      for (const auto& op_input : flat->post_op_inputs) {
        if (!is_safe(op_input.access)) {
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
      for (auto& op_input : flat->post_op_inputs) {
        fixup(op_input.access);
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
                            size_t tile_trials, const VarRewrites& var_rewrites, const TileOptimizer& optimizer,
                            std::map<std::string, KernelInfo>* flat_cache) {
  if (!flat.generate_contraction && !flat.post_ops.size()) {
    // The kernel consists entirely of elided elementwise operations; nothing to do.
    return;
  }
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
  for (auto vec_size = settings.vec_size; flat.agg_vec == 1 && 1 < vec_size; vec_size /= 2) {
    flat = Vectorize(flat, vec_size);
  }
  std::string flat_key = flat.CacheKeyString(vars);
  auto it = flat_cache->find(flat_key);
  if (it != flat_cache->end()) {
    IVLOG(2, "Cache key: " << flat_key << ", Hit!");
    r.kernels.emplace_back(it->second);
    auto& ki = r.kernels.back();
    ki.outputs = flat.kernel_outputs;
    ki.inputs.clear();
    for (const auto& input : inputs) {
      if (vars.at(input).tag == Binding::TENSOR) {
        ki.inputs.emplace_back(var_rewrites.Lookup(input));
      }
    }
    for (const auto& op_input : flat.post_op_inputs) {
      ki.inputs.emplace_back(var_rewrites.Lookup(op_input.name));
    }
    return;
  }
  IVLOG(2, "Cache key: " << flat_key << ", Miss!");

  IVLOG(4, "Optimizing " << kname);
  auto options = optimizer.OptionsFor(kname, settings, flat, tile_trials);

  KernelInfo primary;
  for (size_t i = 0; i < options.size(); i++) {
    const auto& option = options[i];
    KernelInfo ki = GenerateContractionKernel(kname, settings, c, flat, option, inputs, vars, var_rewrites);
    if (i == 0) {
      primary = ki;
    } else {
      primary.candidates.push_back(ki);
    }
  }
  flat_cache->emplace(flat_key, primary);
  r.kernels.emplace_back(std::move(primary));
}

static bool DifferentSize(const Binding& a, const Binding& b) {
  if (a.tag != Binding::TENSOR || b.tag != Binding::TENSOR) {
    return true;
  }
  return a.shape.elem_size() != b.shape.elem_size();
}

static bool SameSizeOrBroadcastCompatible(const Binding& input, const Binding& output) {
  if (input.shape.elem_size() == output.shape.elem_size()) {
    return true;
  }
  if (output.shape.dims.size() < input.shape.dims.size()) {
    return false;
  }
  size_t off = output.shape.dims.size() - input.shape.dims.size();
  for (size_t i = 0; i < input.shape.dims.size(); i++, off++) {
    if (input.shape.dims[i].size != 1 && input.shape.dims[i].size != output.shape.dims[off].size) {
      return false;
    }
  }
  return true;
}

static bool CanUnifyOp(const Program& prog, const Bindings& vars, std::size_t root_opidx, std::size_t test_opidx) {
  const Op& root_op = prog.ops[root_opidx];
  const Op& test_op = prog.ops[test_opidx];
  IVLOG(4, "Testing for unification: " << root_op << " with " << test_op);
  if (test_op.tag != Op::FUNCTION || test_op.f.is_special()) {
    IVLOG(4, "  Downstream is not a simple elementwise operation");
    return false;
  }

  if (DifferentSize(vars.at(root_op.output), vars.at(test_op.output))) {
    IVLOG(4, "  Var " << root_op.output << " differs in size from " << test_op.output);
    return false;
  }

  for (const auto& input : test_op.inputs) {
    if (vars.at(input).tag != Binding::TENSOR) {
      continue;
    }
    if (!SameSizeOrBroadcastCompatible(vars.at(input), vars.at(root_op.output))) {
      // This input requires broadcasting, but it's not
      // dimensionally compatible with the kernel output shape;
      // there's a reshape involved, making it tricky to read from
      // within a kernel output loop.  So we can't use this
      // operation.
      IVLOG(4, "  Input " << input << " is incompatible with the output shape");
      return false;
    }
  }

  IVLOG(4, "  LGTM");
  return true;
}

static void ConsiderConsumers(const Program& prog, const Bindings& vars, std::size_t root_opidx,
                              const std::set<size_t>& previously_computed, const UseDef& ud, std::set<size_t>* unified,
                              std::stack<std::string>* unified_frontier, std::set<std::string>* seen_vars,
                              const std::string& var) {
  // This function is used by ConnectedComponents to extend the connected components subgraph
  // from a particular variable, attempting to add downstream operations whose other inputs'
  // producers have either already been unified or that can be unified into the current
  // kernel being built.  See ConnectedComponents for a broader overview of the algorithm.

  // Loop over the variable's consumers.
  for (std::size_t c_start : ud.uses().at(var)) {
    if (unified->count(c_start) || !CanUnifyOp(prog, vars, root_opidx, c_start) || previously_computed.count(c_start)) {
      continue;
    }

    std::set<std::size_t> candidates;
    std::stack<std::size_t> candidate_frontier;

    candidates.insert(c_start);
    candidate_frontier.push(c_start);

    while (!candidate_frontier.empty()) {
      size_t c = candidate_frontier.top();
      candidate_frontier.pop();

      for (const std::string& input : prog.ops[c].inputs) {
        auto it = ud.op_defs().find(input);
        if (it == ud.op_defs().end()) {
          continue;
        }
        size_t i = it->second;
        if (i < root_opidx || unified->count(i) || candidates.count(i) || previously_computed.count(i)) {
          continue;
        }
        auto tag = prog.ops[i].tag;
        if (tag == Op::CONSTANT) {
          continue;
        }
        if (!CanUnifyOp(prog, vars, root_opidx, i)) {
          goto discard_candidate_set;
        }
        candidates.insert(i);
        candidate_frontier.push(i);
      }
    }

#ifdef __APPLE__
    // HACK: this is to avoid limitations in the number of arguments allowed to a kernel under Metal.
    if (unified->size() + candidates.size() > 10) {
      goto discard_candidate_set;
    }
#endif

    unified->insert(candidates.begin(), candidates.end());
    for (auto c : candidates) {
      for (const auto& var : prog.ops[c].inputs) {
        if (vars.at(var).tag != Binding::TENSOR) {
          continue;
        }
        if (seen_vars->emplace(var).second) {
          unified_frontier->push(var);
        }
      }
      // By definition, we've never seen the current op's output.
      seen_vars->emplace(prog.ops[c].output);
      unified_frontier->push(prog.ops[c].output);
    }

  discard_candidate_set : {}
  }
}

static std::set<size_t> ConnectedComponents(const Program& prog, const Bindings& vars, std::size_t root_opidx,
                                            const std::set<size_t>& previously_computed, const UseDef& ud) {
  // This function computes the set of function operations that can be unified with the indicated initial operation,
  // 'start'.
  //
  // The algorithm is relatively simplistic.  You could imagine unifying function ops with contractions, pushing the
  // starting op forward (so that more subsequent ops can unify with it), or even evaluating function ops multiple times
  // instead of exactly once, which may in some cases allow us to save some intermediate memory -- and perhaps at some
  // point we will implement optimizations like that, but not today.
  //
  // The current implementation starts with the constraint that the starting op will be issued in its existing sequence
  // with all other contraction ops.  The goal of the unification algorithm is simply to determine the set of future
  // function ops that can be unified with the initial function op.
  //
  // Unification is performed iff:
  //
  //   1) Either:
  //      A - The downstream op takes as an input one of the products of the current set's outputs
  //      B - The downstream op produces an output that enables another op to become part of the current set
  //
  //   2) The downstream op's inputs are available at the point where the starting op is issued
  //
  // The algorithm tracks a frontier of function ops to process; this is always a subset of the final op set.  For the
  // current frontier op being processed, each consumer of the current op's output is considered as a candidate for
  // inclusion (automatically
  // satisfying condition 1.A).  If the candidate's inputs are available (either coming from operations issued before
  // start, or coming from operations that're already part of the set), condition 2 is satisfied, and the candidate is
  // added to the set of ops to be unified, as well as to the frontier.
  //
  // To satisfy 1.B, when the candidate might be unifiable if a unifiable parent were included, we consider each
  // candidate as a set of candidates, built by tracing the inputs of each op in the candidate set.  The candidate set
  // is either added as a whole or discarded.
  //
  // We process each frontier depth-first in order to slightly increase memory locality, although at this scale, it
  // doesn't matter much.
  std::set<size_t> unified;
  std::stack<std::string> unified_frontier;
  std::set<std::string> seen_vars;

  // The root operation is always unified.
  unified.insert(root_opidx);

  // Explore unification given the existence of the root operation's inputs and output in the kernel.
  // This recursively adds all consumers of these vars and their respective inputs (iff those inputs can be unified)
  // to the unified set.
  for (const auto& var : prog.ops[root_opidx].inputs) {
    if (vars.at(var).tag != Binding::TENSOR) {
      continue;
    }
    seen_vars.emplace(var);
    unified_frontier.push(var);
  }
  seen_vars.emplace(prog.ops[root_opidx].output);
  unified_frontier.push(prog.ops[root_opidx].output);
  while (!unified_frontier.empty()) {
    std::string var = std::move(unified_frontier.top());
    unified_frontier.pop();
    ConsiderConsumers(prog, vars, root_opidx, previously_computed, ud, &unified, &unified_frontier, &seen_vars, var);
  }

  return unified;
}

static void DoUnification(FlatContraction* flat, std::set<std::size_t>* computed, VarRewrites* var_rewrites,
                          const Program& prog, std::size_t opidx, const UseDef& ud, const Bindings& vars,
                          const ShapeMap& inputs, const ShapeMap& outputs, const std::vector<Polynomial>& out_poly,
                          const HardwareSettings& settings) {
  // Unify the contraction with downstream elementwise operations.
  //
  // Here's the idea: during a contraction's output phase, we
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
  const TensorShape& out_shape = vars.at(flat->output).shape;

  // Additional inputs required for the unified kernel.
  std::set<std::string> post_contraction_set;
  std::vector<std::string> post_contraction_inputs;

  // The variable remappings that have been made in the current
  // kernel.  When talking about a kernel's input parameters, we use
  // original variable names, so that shape lookups are correct.  For
  // locals generated within a kernel, when we encounter a reshape or
  // ident operation, we elide the operation, and replace elementwise
  // inputs with the source variable names.  This just makes the
  // generated code slightly cleaner; alternatives would be to only
  // emit the reshape/ident variables when they're used (slightly
  // trickier), or to always leave them in the generated code (which
  // looks like a mistake when you're reading the code), or to elide
  // them later iff unused (again, trickier).
  std::unordered_map<std::string, std::string> local_var_rewrites;

  // The initial set of inputs supplied to the kernel.
  std::set<std::string> kernel_inputs{op.inputs.begin(), op.inputs.end()};

  IVLOG(3, "In unification, out polys = " << out_poly);

  // The set of elementwise operations that have been unified with the kernel.
  std::set<std::size_t> unified_opidxs = ConnectedComponents(prog, vars, opidx, *computed, ud);

  // The map of outputs that are allowed to alias inputs.
  std::map<std::string, std::set<std::string>> aliases;

  for (auto unified_opidx : unified_opidxs) {
    auto& unified_op = prog.ops[unified_opidx];

    if (unified_op.tag != Op::FUNCTION) {
      continue;
    }

    // Attempt to elide reshape and ident operations.
    //
    // Note that there are several interesting cases here:
    //
    // * If both pre- and post-variables are program outputs, we actually need to write both -- this is a little
    // pointless, but it's valid.  So we keep the reshape or ident operation.
    //
    // * If pre- is a program input, and post- is a program output, we need to copy the input.  So again, we keep the
    // reshape or ident operation.
    //
    // * Otherwise, we can elide the reshape or ident, and use either name for the variable.  We choose to preserve the
    // pre-variable name, map the post-name to the pre-name in subsequent kernels and in the program output bindings,
    // and elide writing the post-variable (although if the post-variable is used downstream, we need to be sure this
    // causes the pre-variable to be written): this may allow subsequent kernels to get started slightly sooner.
    if (unified_op.f.fn == "reshape" || unified_op.f.fn == "ident") {
      if (unified_op.inputs.size() < 1) {
        throw std::runtime_error("reshape must have at least one parameter");
      }
      const auto& in_binding = vars.at(unified_op.inputs[0]);
      const auto& out_binding = vars.at(unified_op.output);
      if (in_binding.tag != Binding::TENSOR) {
        throw std::runtime_error("reshape only works on tensors");
      }
      if (in_binding.shape.byte_size() != out_binding.shape.byte_size()) {
        IVLOG(1, "Input shape = " << in_binding.shape);
        IVLOG(1, "Output shape = " << out_binding.shape);
        throw std::runtime_error("Invalid reshape");
      }
      if (in_binding.shape.elem_size() != out_binding.shape.elem_size()) {
        IVLOG(1, "Input shape = " << in_binding.shape);
        IVLOG(1, "Output shape = " << out_binding.shape);
        throw std::runtime_error("Invalid reshape");
      }

      std::string input;
      input = var_rewrites->Lookup(unified_op.inputs[0]);
      if (!outputs.count(unified_op.output) || (!outputs.count(input) && !inputs.count(input))) {
        IVLOG(4, "  Eliding op:" << unified_op << "; replacing " << unified_op.output << " with " << input);
        var_rewrites->Insert(unified_op.output, input);
        local_var_rewrites.emplace(unified_op.output, std::move(input));
        continue;
      } else {
        IVLOG(4, "  Keeping reshape/ident op:" << unified_op);
      }
    }

    IVLOG(4, "  Unifying op " << unified_op);

    // Adjust inputs to account for local variable rewrites, and add them to the the post-contraction
    // inputs if needed; for each input that's compatible with the output shape (all dimension sizes
    // and striding identical, and identical element size), mark that the output is allowed to alias
    // that input, recursively.
    Op copied_op = unified_op;
    const auto* oshape = &vars.at(copied_op.output).shape;
    for (std::string& input : copied_op.inputs) {
      //// For each input compute the mapping from tensor indexes to global indexes
      const TensorShape* shape = &vars.at(input).shape;
      if (shape->elem_size() == out_shape.elem_size()) {
        shape = &out_shape;
      }
      std::vector<Polynomial> indexes;
      size_t off = out_poly.size() - shape->dims.size();
      for (size_t i = 0; i < shape->dims.size(); i++, off++) {
        indexes.push_back(out_poly[off]);
      }
      flat->index_mapping.emplace(input, indexes);
      IVLOG(4, "Adding mapping for " << input << "=" << indexes);

      auto rit = local_var_rewrites.find(input);
      if (rit != local_var_rewrites.end()) {
        input = rit->second;
      }

      auto uit = ud.op_defs().find(input);
      if (vars.at(input).tag == Binding::TENSOR &&
          (uit == ud.op_defs().end() || (!unified_opidxs.count(uit->second)))) {
        if (!post_contraction_set.count(input)) {
          post_contraction_set.insert(input);
          post_contraction_inputs.push_back(input);
        }
      }

      const auto* ishape = &vars.at(input).shape;
      if (!settings.disable_io_aliasing && ishape->dims == oshape->dims &&
          byte_width(ishape->type) == byte_width(oshape->type)) {
        aliases[copied_op.output].emplace(input);
        auto it = aliases.find(input);
        if (it != aliases.end()) {
          aliases[copied_op.output].insert(it->second.begin(), it->second.end());
        }
      }
    }

    flat->post_ops.emplace_back(std::move(copied_op));
  }

  // For all available outputs: if the usedefs or program outputs
  // require it, add it to the kernel outputs.  Reshaped/identity
  // outputs are never added to the kernel outputs, but if they're
  // needed downstream, they do cause their pre-reshape variables to
  // be emitted as outputs.
  std::set<std::string> kernel_outputs;
  for (auto unified_opidx : unified_opidxs) {
    auto& unified_op = prog.ops[unified_opidx];
    if (kernel_inputs.count(var_rewrites->Lookup(unified_op.output))) {
      // This was a kernel input; it never needs to be a kernel output.
      continue;
    }
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
      kernel_outputs.insert(var_rewrites->Lookup(unified_op.output));
    }
  }

  flat->kernel_outputs.insert(flat->kernel_outputs.end(), kernel_outputs.begin(), kernel_outputs.end());
  if (!settings.disable_io_aliasing) {
    for (const std::string& output : flat->kernel_outputs) {
      auto ait = aliases.find(output);
      if (ait != aliases.end()) {
        for (auto it = ait->second.begin(); it != ait->second.end();) {
          auto eit = it++;
          if (!post_contraction_set.count(*eit)) {
            ait->second.erase(eit);
          }
        }
        flat->safe_self_aliases.emplace(output, std::move(ait->second));
      }
    }
  }

  // Copy over post contraction inputs and compute strides
  computed->insert(unified_opidxs.begin(), unified_opidxs.end());
  for (const auto& name : post_contraction_inputs) {
    const TensorShape* shape = &vars.at(name).shape;
    if (shape->elem_size() == out_shape.elem_size()) {
      // Special case for when the post-contraction input has the same
      // number of elements as the operation output: we use the
      // operation output shape.
      //
      // This allows us to correctly handle contractionless kernels
      // whose first operation is a reshape, and kernels that include
      // a reshape and post-reshape elementwise operations that don't
      // involve broadcasts.
      //
      // In those cases, the post-contraction input may be an
      // arbitrary shape, which makes it impossible to derive a
      // FlatTensorAccess that's compatible with the overall output of
      // the kernel.  Since the element count is identical, it's safe
      // to go ahead and use the output shape; the accesses will have
      // no connection to the actual shape of the input, but for
      // elementwise operations that's completely fine.
      //
      // (Note that we carefully filter out elements whose inputs are
      // not broadcast-compatible with the overall kernel output
      // shape.  Handling these correctly is non-trivial, since we'd
      // need to build the shape of each elementwise operation and
      // read the broadcasted input based on that.  It's certainly not
      // impossible to do so, though.)
      shape = &out_shape;
    }
    FlatTensorAccess access;
    access.global_index_limit = shape->elem_size();
    Polynomial p;
    size_t off = out_poly.size() - shape->dims.size();
    for (size_t i = 0; i < shape->dims.size(); i++, off++) {
      // We add things if they are not broadcast, we treat 1, 1 as non broadcast in this case
      if (shape->dims[i].size != 1 || out_shape.dims[off].size == 1) {
        p += out_poly[off] * shape->dims[i].stride;
      }
    }
    for (const auto& idx : flat->names) {
      access.strides.push_back(static_cast<int64_t>(Floor(p[idx])));
    }
    auto binding = vars.at(name);
    access.type = binding.shape.type;
    IVLOG(3, "For shape: " << shape << " poly = " << p << " strides = " << access.strides);
    FlatPostOpInput post_op_input = {name, access, binding};
    flat->post_op_inputs.emplace_back(post_op_input);
  }
}

static KernelList Compile(const Program& orig_prog, const ShapeMap& inputs, const ShapeMap& outputs,
                          const HardwareSettings& settings, const std::string& kid, size_t tile_trials,
                          const TileOptimizer& optimizer) {
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
  std::map<std::string, KernelInfo> flat_cache;
  for (size_t i = 0; i < prog.ops.size(); i++) {
    if (time(nullptr) - last_update >= 2) {
      LOG(INFO) << "Analyzing Ops: " << i << " of " << prog.ops.size() << " operations complete";
      last_update = time(nullptr);
    }
    const Op& op = prog.ops[i];

    if (op.tag == Op::CONTRACTION) {
      IVLOG(3, "Running contraction " << op << " vars = " << vars);
      if (vars.at(op.output).shape.byte_size() == 0) {
        IVLOG(3, "Contraction output " << op.output << " size==0; skipping");
        continue;
      }
      std::vector<TensorShape> tshapes = MakeTShapes(op.c, vars);
      std::vector<Polynomial> out_poly;
      FlatContraction flat = Compile(op.c, tshapes, &out_poly);
      flat.output = op.output;

      auto kname = next_kname();
      if (NeedsZero(flat, tshapes[0])) {
        // N.B. We currently don't unify kernels with subsequent
        // operations unless they cover the entire output space.
        if (op.c.use_default != "") {
          r.kernels.push_back(GenCopy(tshapes[0], op.output, op.c.use_default, "copy_" + kname));
        } else {
          r.kernels.push_back(GenZero(tshapes[0], op.output, "zero_" + kname));
        }
        flat.kernel_outputs.push_back(op.output);
      } else {
        DoUnification(&flat, &computed, &r.var_rewrites, prog, i, ud, vars, inputs, outputs, out_poly, settings);
      }
      ContractionWrap(r, &op.c, std::move(flat), kname, settings, vars, tile_trials, r.var_rewrites, optimizer,
                      &flat_cache);
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
          throw std::runtime_error("prng_step function missing its companions");
        }
        dop.f.params.push_back(sout);
        dop.f.params.push_back(vout);
      }
      if (op.f.fn == "scatter") {
        r.kernels.push_back(GenZero(vars.at(op.output).shape, op.output, "zero_" + next_kname()));
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

    flat.comb_op = CombinationOp::NONE;
    flat.agg_op = AggregationOp::NONE;

    std::vector<Polynomial> out_poly;
    {
      // The initial elementwise operation's output is used to
      // determine the shape of the overall kernel -- which is
      // reasonable, because every subsequent elementwise operation is
      // required to have an output that's same shape as that initial
      // operation.
      flat.generate_contraction = false;

      const auto& access_op = prog.ops[i];

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
      access.global_index_limit = shape.elem_size();
      for (const auto& dim : shape.dims) {
        access.strides.emplace_back(dim.stride);
      }
      flat.access.emplace_back(std::move(access));
    }

    DoUnification(&flat, &computed, &r.var_rewrites, prog, i, ud, vars, inputs, outputs, out_poly, settings);

    ContractionWrap(r, nullptr, std::move(flat), next_kname(), settings, vars, tile_trials, r.var_rewrites, optimizer,
                    &flat_cache);
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
                           const HardwareSettings& settings, const TileOptimizer& optimizer, const std::string& id,
                           size_t tile_trials) {
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
  KernelList result;
  result = Compile(prog, inputs, outputs, settings, kid, tile_trials, optimizer);
  Simplify(result.kernels);
  return result;
}

void TileOptimizer::RegisterModel(const TileCostFunction& cost_fn) { models_.push_back(cost_fn); }

TileOptions TileOptimizer::OptionsFor(const std::string& kname, const HardwareSettings& settings,
                                      const FlatContraction& op, size_t max_options) const {
  TileOptions options;
  if (models_.empty()) {
    auto by_score = TileOptimize(settings, op, max_options == 1);
    size_t count = 0;
    for (auto it = by_score.rbegin(); it != by_score.rend() && count < max_options; it++, count++) {
      options.emplace_back(TileOption{"", it->second, it->first, it->first, it->first});
    }
  } else {
    std::multimap<double, TileOption> by_cost;
    for (const auto& model : models_) {
      auto model_options = model(kname, settings, op);
      for (const auto& option : model_options) {
        by_cost.insert(std::make_pair(option.kernel_cost, option));
      }
    }
    size_t count = 0;
    for (auto it = by_cost.begin(); it != by_cost.end() && count < max_options; it++, count++) {
      VLOG(1) << "Option: " << it->second.model;
      options.push_back(it->second);
    }
  }
  return options;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
