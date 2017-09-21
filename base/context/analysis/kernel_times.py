from __future__ import print_function

import argparse

import base.context.analysis as ca

from tile.hal.opencl import opencl_pb2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display kernel times')
    parser.add_argument('file', help='a binary profile data file')
    args = parser.parse_args()
    scope = ca.Scope()
    scope.read_eventlog(args.file)
    for id, evt in scope.events.iteritems():
        if evt.verb == 'tile::hal::opencl::Kernel::Executing':
            try:
                kname = evt.first_metadatum(opencl_pb2.RunInfo).kname
            except KeyError:
                kname = '<unknown>'
            delta = evt.end_time - evt.start_time
            print('%s %d' % (kname, delta.seconds * 1000000 + delta.microseconds))
