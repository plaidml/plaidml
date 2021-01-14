
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

Operation *findDeallocPair(Operation *alloc) {
  if (!alloc)
    return nullptr;
  // Not an alloc if it returns multiple results
  if (alloc->getNumResults() != 1)
    return nullptr;
  // Not an alloc if it doesn't implement memory effect intereface
  auto memEff = dyn_cast<MemoryEffectOpInterface>(alloc);
  if (!memEff)
    return nullptr;
  Value val = alloc->getResult(0);
  SmallVector<MemoryEffects::EffectInstance, 2> effects;
  memEff.getEffectsOnValue(val, effects);
  // Not an alloc if it doesn't have a single effect of MemAlloc
  if (effects.size() != 1)
    return nullptr;
  if (!isa<MemoryEffects::Allocate>(effects[0].getEffect()))
    return nullptr;
  // Start search for dealloc
  Operation *dealloc = nullptr;
  for (auto *user : val.getUsers()) {
    // Not a dealloc if it doens't support memory effect interface
    auto userMemEff = dyn_cast<MemoryEffectOpInterface>(user);
    if (!userMemEff)
      continue;
    SmallVector<MemoryEffects::EffectInstance, 2> userEffects;
    userMemEff.getEffectsOnValue(val, userEffects);
    // Not a dealloc if it doesn't have a unique Free effect
    if (userEffects.size() != 1)
      continue;
    if (!isa<MemoryEffects::Free>(userEffects[0].getEffect()))
      continue;
    // If we find more than one dealloc, fail
    if (dealloc != nullptr)
      return nullptr;
    dealloc = user;
  }
  // Return the unique dealloc we found (if any)
  return dealloc;
}

} // namespace pmlc::util
