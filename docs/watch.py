#!/usr/bin/env python

from livereload import Server, shell

doxygen = shell('doxygen', cwd='docs')

sphinx = shell('sphinx-build -M html . _build', cwd='docs')


def rebuild():
    doxygen()
    sphinx()


rebuild()

server = Server()
server.watch('plaidml2/', rebuild)
server.watch('docs/conf.py', sphinx)
server.watch('docs/**/*.rst', sphinx)
server.serve(root='docs/_build/html')
