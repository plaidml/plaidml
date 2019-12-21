#!/usr/bin/env python

from livereload import Server, shell

build = shell('naturaldocs docs/natural')
build()

server = Server()
server.watch('docs/natural/*.txt', build)
server.watch('plaidml2/', build)
server.serve(root='docs/html')
