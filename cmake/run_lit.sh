#!/bin/bash

# Copyright 2019 Google LLC
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
set -e
set -o pipefail

if [ -z "${RUNFILES_DIR}" ]; then
  # Some versions of bazel do not set RUNFILES_DIR. Instead they just cd
  # into the directory.
  RUNFILES_DIR="$PWD"
fi

# Detect whether cygwin/msys2 paths need to be translated.
set +e  # Ignore errors if not found.
cygpath="$(which cygpath 2>/dev/null)"
set -e

function find_executables() {
  set -e
  local p="$1"
  if [ "$(uname)" == "Darwin" ]; then
    # For macOS, xtype isn't avaliable and perm can't use `/u=x,g=x,o=x` syntax.
    find "${p}" -type l -perm +111
  elif [ -z "$cygpath" ]; then
    # For non-windows, use the perm based executable check, which has been
    # supported by find for a very long time.
    find "${p}" -xtype f -perm /u=x,g=x,o=x -print
  else
    # For windows, always use the newer -executable find predicate (which is
    # not supported by ancient versions of find).
    find "${p}" -xtype f -executable -print
  fi
}

# Bazel helpfully puts all data deps in the ${RUNFILES_DIR}, but
# it unhelpfully preserves the nesting with no way to reason about
# it generically. run_lit expects that anything passed in the runfiles
# can be found on the path for execution. So we just iterate over the
# entries in the MANIFEST and extend the PATH.
SUBPATH=""
for runfile_path in $(find_executables "${RUNFILES_DIR}"); do
  # Prepend so that local things override.
  EXEDIR="$(dirname ${runfile_path})"
  if ! [ -z "$cygpath" ]; then
    EXEDIR="$($cygpath -u "$EXEDIR")"
  fi
  SUBPATH="${EXEDIR}:$SUBPATH"
done

echo "run_lit.sh: $1"
echo "PWD=$(pwd)"

# For each "// RUN:" line, run the command.
runline_matches="$(egrep "^// RUN: " "$1")"
if [ -z "$runline_matches" ]; then
  echo "!!! No RUN lines found in test"
  exit 1
fi

echo "$runline_matches" | while read -r runline
do
  echo "RUNLINE: $runline"
  match="${runline%%// RUN: *}"
  command="${runline##// RUN: }"
  if [ -z "${command}" ]; then
    echo "ERROR: Could not extract command from runline"
    exit 1
  fi

  # Substitute any embedded '%s' with the file name.
  full_command="${command//\%s/$1}"

  # Run it.
  export PATH="$SUBPATH:$PATH"
  echo "RUNNING TEST: $full_command"
  echo "----------------"
  if eval "$full_command"; then
    echo "--- COMPLETE ---"
  else
    echo "!!! ERROR EVALUATING: $full_command"
    exit 1
  fi
done
