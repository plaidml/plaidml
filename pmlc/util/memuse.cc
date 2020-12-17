
#include "pmlc/util/memuse.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir; // NOLINT

namespace pmlc::util {

MemUse getMemoryUses(Value def) {
  uint64_t useBits = 0;
  for (auto *user : def.getUsers()) {
    auto userMemInterface = dyn_cast<MemoryEffectOpInterface>(user);
    if (!userMemInterface) {
      continue;
    }
    SmallVector<MemoryEffects::EffectInstance, 2> userEffects;
    userMemInterface.getEffectsOnValue(def, userEffects);
    for (auto &userEffectInstance : userEffects) {
      auto *userEffect = userEffectInstance.getEffect();
      if (userEffect && isa<MemoryEffects::Read>(userEffect)) {
        useBits |= static_cast<uint64_t>(util::MemUse::read_only);
      }
      if (userEffect && isa<MemoryEffects::Write>(userEffect)) {
        useBits |= static_cast<uint64_t>(util::MemUse::write_only);
      }
    }
  }
  return static_cast<util::MemUse>(useBits);
}

} // namespace pmlc::util
