// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/util/selector.h"

#include <boost/regex.hpp>

namespace vertexai {
namespace tile {
namespace hal {
namespace selector {

bool Match(const proto::HardwareSelector& sel, const proto::HardwareInfo& info, std::uint_fast32_t depth_allowed) {
  if (!depth_allowed) {
    return false;
  }
  switch (sel.selector_case()) {
    case proto::HardwareSelector::kValue:
      return sel.value();

    case proto::HardwareSelector::kAnd: {
      bool result = true;
      for (const auto& term : sel.and_().sel()) {
        result &= Match(term, info, depth_allowed - 1);
        if (!result) {
          break;
        }
      }
      return result;
    }

    case proto::HardwareSelector::kOr: {
      bool result = false;
      for (const auto& term : sel.or_().sel()) {
        result |= Match(term, info, depth_allowed - 1);
        if (result) {
          break;
        }
      }
      return result;
    }

    case proto::HardwareSelector::kNot:
      return !Match(sel.not_(), info, depth_allowed - 1);

    case proto::HardwareSelector::kType:
      return sel.type() == info.type();

    case proto::HardwareSelector::kNameRegex:
      return boost::regex_match(info.name(), boost::regex(sel.name_regex()));

    case proto::HardwareSelector::kVendorRegex:
      return boost::regex_match(info.vendor(), boost::regex(sel.vendor_regex()));

    case proto::HardwareSelector::kPlatformRegex:
      return boost::regex_match(info.platform(), boost::regex(sel.platform_regex()));

    case proto::HardwareSelector::kVendorId:
      return sel.vendor_id() == info.vendor_id();

    default:
      return false;
  }
}

}  // namespace selector
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
