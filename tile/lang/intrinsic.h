#pragma once

#include <map>
#include <string>

#include "tile/lang/semtree.h"
#include "tile/lang/type.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace lang {

// The emitter for intrinsic
// Parameters: name, inputs, outputs
typedef sem::ExprPtr (*IntrinsicEmitter)(const stripe::Intrinsic&);

struct IntrinsicSpec {
  sem::ExprPtr emit(const stripe::Intrinsic& stmt) const { return emitter_(stmt); }

  std::string name_;
  IntrinsicEmitter emitter_;
};

class IntrinsicList {
 public:
  IntrinsicList() : default_emitter_(nullptr) {}
  explicit IntrinsicList(const IntrinsicEmitter& emitter) : default_emitter_(emitter) {}
  void add(const IntrinsicSpec& spec) { map_.emplace(spec.name_, spec); }
  bool exist(const std::string& name) const { return map_.find(name) != map_.end(); }
  sem::ExprPtr emit(const stripe::Intrinsic& in) const;

 private:
  std::map<std::string, IntrinsicSpec> map_;
  IntrinsicEmitter default_emitter_;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
