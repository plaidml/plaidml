#include "tile/codegen/access.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

bool operator==(const AccessPattern& lhs, const AccessPattern& rhs) {
  return std::tie(lhs.is_write, lhs.is_exact, lhs.idxs, lhs.access, lhs.constraints) ==
         std::tie(rhs.is_write, rhs.is_exact, rhs.idxs, rhs.access, rhs.constraints);
}

// Per index information
struct IdxInfo {
  // Is the index incomplete?  That is, does it derive from an index we are not tracking?
  bool incomplete = false;
  // What is the global value of the index (presuming it's not incomplete)
  std::vector<int64_t> gid;
};

struct Context {
  size_t idx_count = 0;                 // How many indexes exist in this + parent contexts?
  bool exact = true;                    // Are we still exact, or did we skip constraints based on incomplete indexes?
  std::vector<stripe::Index> idxs;      // The current set of indexes
  std::vector<Constraint> constraints;  // Current constraints
  std::map<std::string, IdxInfo> idx_info;     // Information on each current index by name
  std::map<std::string, BufferAccess> access;  // The information on each in scope buffer
};

void ComputeAccessRecursive(std::vector<AccessPattern>* out, const Block& block, const Context& up) {
  // Prepare the new context
  Context self;
  // Update count of indexes
  self.idx_count = up.idx_count + block.idxs.size();
  // Copy across exactness
  self.exact = up.exact;
  // Copy across existing indexes
  self.idxs = up.idxs;
  // Copy across any constraints
  self.constraints = up.constraints;
  // First process all refinements
  for (const auto& ref : block.refs) {
    // Check if source is in our parent
    auto it = up.access.find(ref.from);
    // If not, don't don't bother
    if (it == up.access.end()) {
      continue;
    }
    // Copy across for a starting point
    BufferAccess access = it->second;
    // Add the additional offset
    access.offset += ref.access.offset;
    // Add the additional strides
    for (const auto& stride : ref.access.strides) {
      access.strides.push_back(stride);
    }
    // Make a new access info for the 'into' name
    self.access[ref.into] = access;
  }
  // If we have no refinements left, early return
  if (self.access.empty()) {
    return;
  }
  // Make index info for each index
  for (const auto& idx : block.idxs) {
    // Get the id of this index
    size_t iid = self.idxs.size();
    // Add the index name + range
    self.idxs.push_back(idx);
    // Make a place to put the info on this index
    IdxInfo& info = self.idx_info[idx.name];
    // Check for the index in the outer context
    auto upit = up.idx_info.find(idx.name);
    if (upit == up.idx_info.end()) {
      // This is a new index as far as we have record, check if it has a parent we didn't know about
      if (idx.factor) {
        // Yup, we can't know about the gid...
        info.incomplete = true;
      } else {
        // Gid is just outselved
        info.gid.resize(self.idx_count);
        info.gid[iid] = 1;
      }
    } else {
      const auto& old_info = upit->second;
      if (old_info.incomplete && idx.factor) {
        // If we derive from an incomplete index, we are also incomplete
        info.incomplete = true;
      } else {
        // Otherwise, multiply previous gid by factor and add self
        info.gid.resize(self.idx_count);
        for (size_t i = 0; i < old_info.gid.size(); i++) {
          info.gid[i] = idx.factor * old_info.gid[i];
        }
        info.gid[iid] = 1;
      }
    }
  }
  // Next add all new constraints
  for (const auto& pcon : block.constraints) {
    Constraint con;
    con.lhs.resize(self.idx_count);
    bool exact = true;
    for (size_t i = 0; i < size_t(pcon.lhs.size()); i++) {
      int64_t mul = pcon.lhs[i];
      const auto& info = self.idx_info[block.idxs[i].name];
      if (info.incomplete && mul != 0) {
        exact = false;
        break;
      }
      const auto& gid = self.idx_info[block.idxs[i].name].gid;
      for (size_t j = 0; j < gid.size(); j++) {
        con.lhs[j] += mul * gid[j];
      }
    }
    con.rhs = pcon.rhs;
    if (exact) {
      self.constraints.push_back(con);
    } else {
      self.exact = false;
    }
  }
  // Make another labmda to construct and add an access pattern
  auto add_access = [&](const std::string& name, bool is_write) {
    auto it = self.access.find(name);
    if (it == self.access.end()) {
      return;
    }
    out->emplace_back(AccessPattern{
        is_write,         // is_write
        self.exact,       // is_exact
        self.idxs,        // idxs
        it->second,       // access
        self.constraints  // constraints
    });
  };
  // Now go over all statements (and possibly recurse)
  for (const auto& stmt : block.stmts) {
    switch (stmt->kind()) {
      case StmtKind::Load:
        add_access(Load::Downcast(stmt)->from, false);
        break;
      case StmtKind::Store:
        add_access(Store::Downcast(stmt)->from, true);
        break;
      case StmtKind::Block:
        // TODO: Consider prequalifying block (does it refine anything I care about) before descending
        ComputeAccessRecursive(out, *Block::Downcast(stmt), self);
        break;
      default:
        break;
    }
  }
}

std::ostream& operator<<(std::ostream& os, const AccessPattern& ap) {
  os << "Access Pattern: (is_write=" << ap.is_write << " exact=" << ap.is_exact << " offset=" << ap.access.offset
     << ") {\n";
  for (size_t i = 0; i < ap.idxs.size(); i++) {
    os << "  " << ap.idxs[i].name << " range=" << ap.idxs[i].range << " stride=" << ap.access.strides[i] << "\n";
  }
  for (const auto& c : ap.constraints) {
    os << "  ";
    for (const auto& v : c.lhs) {
      os << v << " ";
    }
    os << "< " << c.rhs << "\n";
  }
  os << "}\n";
  return os;
}

std::vector<AccessPattern> ComputeAccess(const Block& block, const std::string& buffer) {
  std::vector<AccessPattern> out;
  Context top;
  top.access[buffer];  // Make an empty access pattern @ named location
  ComputeAccessRecursive(&out, block, top);
  return out;
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
