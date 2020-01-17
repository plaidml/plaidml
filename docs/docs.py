#!/usr/bin/env python

import glob

from livereload import Server, shell

import util

doxygen = shell('doxygen', cwd='docs')
sphinx = shell('sphinx-build -M html . _build', cwd='docs')


def rebuild():
    doxygen()
    util.fix_doxyxml('docs/xml/*.xml')
    sphinx()


def main():
    rebuild()
    server = Server()
    server.watch('plaidml/', rebuild)
    server.watch('docs/Doxyfile', rebuild)
    server.watch('docs/conf.py', sphinx)
    server.watch('docs/**/*.rst', sphinx)
    server.serve(root='docs/_build/html')


if __name__ == "__main__":
    main()
