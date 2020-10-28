#!/bin/bash

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A wrapper around a test command that performs setup and teardown. This is
# appranetly not supported natively in ctest/cmake.

set -x
set -e

function cleanup() {
  echo "Cleaning up test environment"
  rm -rf ${TEST_TMPDIR?}
}

echo "Creating test environment"
rm -rf "${TEST_TMPDIR?}" # In case this wasn't cleaned up previously
mkdir "${TEST_TMPDIR?}"
trap cleanup EXIT
# Execute whatever we were passed.
"$@"
