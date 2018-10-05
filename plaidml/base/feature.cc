// Copyright 2018 Intel Corporation.

#include "plaidml/base/base.h"
#include "plaidml/base/status.h"
#include "plaidml/base/status_strings.h"

// vai_query_feature

extern "C" void* vai_query_feature(vai_feature_id /* id */) {
  vertexai::SetLastStatus(VAI_STATUS_NOT_FOUND, vertexai::status_strings::kNoSuchFeature);
  return nullptr;
}
