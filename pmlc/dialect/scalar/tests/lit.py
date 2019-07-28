#!/usr/bin/env python3

import os
import subprocess
import sys

lit = os.path.join('external', 'llvm', 'lit')
sys.exit(subprocess.run([lit, os.path.dirname(__file__), '-v'] + sys.argv[1:]).returncode)
