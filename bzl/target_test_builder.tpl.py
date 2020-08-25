#!/usr/bin/env python
#
# This script is used to run target-specific bazel tests outside of bazel.

import os
import sys
import tarfile


if len(sys.argv) != 3:
    print('Usage: device_test_builder <output tar file name> <test runner>')

# Use the script name to find the runfiles directory.
#
# N.B. This appears to be a standard mechanism for bazel tools, since
# they're not passed either of the usual environment variables
# (RUNFILES_DIR or RUNFILES_MANIFEST_PATH) that are passed when using
# `bazel run` or `bazel test` to invoke a program.
runfiles = sys.argv[0] + '.runfiles'

archive_path = sys.argv[1]
archive_name = os.path.basename(archive_path)
archive_base = archive_name.split('.', 1)[0]

with tarfile.open(archive_path, 'w:gz') as archive:
    # Add the test runner
    runner = os.path.realpath(sys.argv[2])
    archive.add(runner, os.path.join(archive_base, "run_tests.py"))

    # Add the contents of runfiles
    os.chdir(runfiles)
    for dirname, dirs, filenames in os.walk('.'):
        dirname = os.path.normpath(dirname)
        for filename in filenames:
            if filename == 'MANIFEST':  # Force RUNFILES_DIR mode
                continue
            relname = os.path.join(dirname, filename)
            relname = os.path.normpath(relname)
            arcname = os.path.join(archive_base, relname)
            archive.add(os.path.realpath(relname), arcname)
