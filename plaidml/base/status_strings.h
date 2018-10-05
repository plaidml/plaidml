// Copyright 2018 Intel Corporation.

#pragma once

namespace vertexai {
namespace status_strings {

// The string used for no-error status messages.
extern const char kOk[];

// The string used for out-of-memory status messages.
extern const char kOom[];

// The string used for internal-error status messages.
extern const char kInternal[];

// The string used for no-such-feature status messages.
extern const char kNoSuchFeature[];

// The string used for generic "cancelled" status messages.
extern const char kCancelled[];

// The string used for "no-devices" status messages.
extern const char kNoDevices[];

// The string used for invalid argument status messages.
extern const char kInvalidArgument[];

}  // namespace status_strings
}  // namespace vertexai
