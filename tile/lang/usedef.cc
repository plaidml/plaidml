#include "tile/lang/usedef.h"

#include <set>
#include <stack>

namespace vertexai {
namespace tile {
namespace lang {

UseDef::UseDef(const Program& prog) {
  for (size_t i = 0; i < prog.inputs.size(); i++) {
    const auto& in = prog.inputs[i];
    if (in_defs_.count(in.name)) {
      throw std::runtime_error("Duplicate input " + in.name);
    }
    in_defs_[in.name] = i;
  }
  for (size_t i = 0; i < prog.ops.size(); i++) {
    if (in_defs_.count(prog.ops[i].output) || op_defs_.count(prog.ops[i].output)) {
      throw std::runtime_error("Variable " + prog.ops[i].output + " redeclared");
    }
    op_defs_[prog.ops[i].output] = i;
    if (prog.ops[i].tag == Op::CONSTANT) {
      continue;
    }
    for (const std::string& v : prog.ops[i].inputs) {
      uses_[v].push_back(i);
    }
  }
}

std::set<size_t> UseDef::ConnectedComponents(const Program& prog, size_t start,
                                             const std::set<size_t>& previously_computed) {
  // This method computes the set of function operations that can be unified with the indicated initial operation,
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
  std::stack<size_t> unified_frontier;

  unified.insert(start);
  unified_frontier.push(start);

  while (!unified_frontier.empty()) {
    size_t u = unified_frontier.top();
    unified_frontier.pop();

    // Loop over the current frontier node's output consumers.
    for (size_t c_start : uses_[prog.ops[u].output]) {
      if (unified.count(c_start) || prog.ops[c_start].tag != Op::FUNCTION || prog.ops[c_start].f.is_special() ||
          previously_computed.count(c_start)) {
        continue;
      }

      std::set<size_t> candidates;
      std::stack<size_t> candidate_frontier;

      candidates.insert(c_start);
      candidate_frontier.push(c_start);

      while (!candidate_frontier.empty()) {
        size_t c = candidate_frontier.top();
        candidate_frontier.pop();

        for (const std::string& input : prog.ops[c].inputs) {
          auto it = op_defs_.find(input);
          if (it == op_defs_.end()) {
            continue;
          }
          size_t i = it->second;
          if (i < start || unified.count(i) || candidates.count(i) || previously_computed.count(i)) {
            continue;
          }
          auto tag = prog.ops[i].tag;
          if (tag == Op::CONSTANT) {
            continue;
          }
          if (tag != Op::FUNCTION || prog.ops[i].f.is_special()) {
            goto discard_candidate_set;
          }
          candidates.insert(i);
          candidate_frontier.push(i);
        }
      }

      unified.insert(candidates.begin(), candidates.end());
      for (auto c : candidates) {
        unified_frontier.push(c);
      }

    discard_candidate_set : {}
    }
  }
  return unified;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
