# Copyright 2018 Intel Corporation.
"""Various general-purpose utility classes."""

import os


class RunfilesDB(object):
    """Translates filenames as indicated by a Bazel MANIFEST."""

    def __init__(self, prefix=""):
        if prefix and prefix[-1] != '/':
            prefix = prefix + '/'
        self._prefix = prefix
        manifest_contents = {}
        manifest_filename = None
        if 'RUNFILES_MANIFEST_FILE' in os.environ:
            manifest_filename = os.environ['RUNFILES_MANIFEST_FILE']
        elif 'RUNFILES_DIR' in os.environ:
            manifest_filename = os.path.join(os.environ['RUNFILES_DIR'], 'MANIFEST')
        if manifest_filename and os.path.exists(manifest_filename):
            with open(manifest_filename) as manifest:
                for line in manifest:
                    (logical, physical) = line.split(' ', 2)
                    manifest_contents[logical] = physical.strip()
        self._manifest = manifest_contents

    def __getitem__(self, logical_filename):
        return self._manifest.get(self._prefix + logical_filename, logical_filename)
