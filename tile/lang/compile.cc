#include "tile/lang/compile.h"

#include <cinttypes>
#include <sstream>

#include "tile/lang/bound.h"
#include "tile/lang/defract.h"
#include "tile/lang/parser.h"
#include "tile/lang/reduce.h"

#include "base/util/logging.h"

namespace vertexai {
namespace tile {
namespace lang {

FlatContraction Compile(const Contraction& c, const std::vector<TensorShape>& shapes,
                        std::vector<Polynomial<Rational>>* out_poly) {
  if (c.specs.size() != 2 && c.specs.size() != 3 && c.specs.size() != 4) {
    throw std::runtime_error("Currently, we only support 1, 2, or 3 element Contractions");
  }
  std::ostringstream cs;
  SVLOG(cs, 3, "Original:\n" << to_string(c).c_str());
  Contraction int_idx_cntrc = ConstrainIndexVarsToInts(c);
  SVLOG(cs, 3, "With Index Variables Made Integral:\n" << to_string(int_idx_cntrc).c_str());
  // Check if we can skip reduce
  bool fancy = false;
  for (const Polynomial<Rational>& p : c.specs[0].spec) {
    if (p.getMap().size() > 2 || (p.getMap().size() == 2 && p.constant() == 0)) {
      fancy = true;
      break;
    }
  }
  Contraction reduced = int_idx_cntrc;
  std::vector<RangeConstraint> cons = GatherConstraints(int_idx_cntrc, shapes);
  SVLOG(cs, 3, "Constraints:" << to_string(cons));
  // Reduce if needed
  if (fancy && !c.no_defract) {
    reduced = ReduceOutputPolynomials(int_idx_cntrc, cons);
    SVLOG(cs, 3, "Reduced:\n" << to_string(reduced));
    cons = GatherConstraints(reduced, shapes);
    SVLOG(cs, 3, "Reduced Constraints:" << to_string(cons));
  }
  MergeParallelConstraints(&cons);
  SVLOG(cs, 3, "Merged Parallel Constraints:" << to_string(cons));
  // Defract if needed (defract does early return if not required)
  Contraction defracted = Defract(reduced, cons);
  SVLOG(cs, 3, "Defracted:\n" << to_string(defracted));
  // Flatten
  if (out_poly) {
    *out_poly = defracted.specs[0].spec;
  }
  FlatContraction flat = Flatten(defracted, shapes);
  SVLOG(cs, 3, "Flattened:\n" << to_string(flat).c_str());
  flat.comments = cs.str();
  return flat;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
