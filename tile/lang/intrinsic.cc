
#include "tile/lang/intrinsic.h"

#include <exception>
#include <map>
#include <memory>
#include <utility>

#include "base/util/logging.h"
#include "tile/lang/sembuilder.h"
#include "tile/lang/semprinter.h"

namespace vertexai {
namespace tile {
namespace lang {

sem::ExprPtr IntrinsicList::emit(const stripe::Intrinsic& in) const {
  auto it = map_.find(in.name);
  if (it == map_.end()) {
    if (default_emitter_ == nullptr) {
      throw std::runtime_error("Cannot find intrinsic " + in.name + " and no default emitter.");
    }
    return default_emitter_(in);
  }
  const IntrinsicSpec& spec = it->second;
  return spec.emit(in);
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
