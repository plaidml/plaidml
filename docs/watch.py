#!/usr/bin/env python

import glob

from livereload import Server, shell

doxygen = shell('doxygen', cwd='docs')

sphinx = shell('sphinx-build -M html . _build', cwd='docs')


def fix_doxyxml():
    for xml in glob.glob("docs/xml/*.xml"):
        with open(xml, "r") as xmlfile:
            orig_lines = xmlfile.readlines()
        with open(xml, "w") as xmlfile:
            for line in orig_lines:
                if "</includes>" not in line:
                    xmlfile.write(line)


def rebuild():
    doxygen()
    fix_doxyxml()
    sphinx()


rebuild()

server = Server()
server.watch('plaidml/', rebuild)
server.watch('docs/conf.py', sphinx)
server.watch('docs/**/*.rst', sphinx)
server.serve(root='docs/_build/html')
