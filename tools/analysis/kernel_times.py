import argparse
import datetime
import os

from asq.initiators import query
from asq.record import new

import tools.analysis as ta


def valid_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError('File not found: {!r}. Use absolute paths.'.format(path))
    return path


def human_size(num):
    if num is None:
        return ''
    fmt = '{:3.1f}{}'
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return fmt.format(num, unit)
        num /= 1024.0
    return fmt.format(num, 'Y')


def main():
    parser = argparse.ArgumentParser(description='Display kernel times')
    parser.add_argument('--file',
                        help='a binary profile data file',
                        type=valid_path,
                        default=os.path.expanduser('~/eventlog.gz'))
    parser.add_argument('--program',
                        help='show program source for kernel in focus',
                        action='store_true')
    parser.add_argument('--ops', help='show ops for each kernel', action='store_true')
    parser.add_argument('--kernel', help='focus on specified kernel, by default slowest')
    parser.add_argument('--dump', help='dump kernels to specified directory', type=valid_path)
    args = parser.parse_args()
    scope = ta.Scope()
    scope.read_eventlog(args.file)

    activities = []
    for act in [x for x in scope.activities if x.verb == 'tile::hal::opencl::Executing']:
        kid = act.ocl_runinfo.kernel_id
        comp = scope.get_activity(act, kid)
        kname = comp.ocl_kernelinfo.kname
        activities.append(new(
            act=act,
            kname=kname,
            comp=comp,
        ))

    def make_result(key, group):
        ktimes = query(group).select(lambda x: x.act.elapsed_time).to_list()
        total = sum(ktimes)
        count = len(ktimes)
        return new(
            kname=key,
            comp=group[0].comp,
            count=count,
            total_runtime=total,
            mean_runtime=(total / count),
        )

    by_runtime = query(activities) \
        .group_by(lambda x: x.kname, result_selector=make_result) \
        .order_by(lambda x: x.mean_runtime) \
        .to_list()

    hdr = '{:24}  {:>6}  {:>12}  {:>12}  {:>7}  {:>7}  {:11}  {:15}'
    fmt = '{:24}  {:>6}  {:12.3f}  {:12.6f}  {:>7}  {:>7}  {:11}  {:15}'
    print(hdr.format('kernel', 'count', 'cumulative', 'self', 'flops/s', 'bytes/s', 'type', 'vec'))
    for item in by_runtime:
        kinfo = item.comp.ocl_kernelinfo.kinfo
        type = kinfo.WhichOneof('kernel_type')
        ops = []
        vec = 'N/A'
        if type == 'contraction':
            ops = kinfo.contraction.ops
            vec = kinfo.contraction.vec
        elif type == 'element':
            ops = kinfo.element.ops
            vec = kinfo.element.vec
        elif type == 'zero':
            if kinfo.zero.copy:
                type = 'copy'
        elif type == 'special':
            type = kinfo.special.fn
        elif type is None:
            type = 'unknown'
        print(
            fmt.format(
                item.kname,
                item.count,
                item.total_runtime,
                item.mean_runtime,
                human_size(kinfo.flops / item.mean_runtime),
                human_size(kinfo.bytes / item.mean_runtime),
                type,
                str(vec),
            ))
        if args.ops:
            for op in ops:
                print(op)
        if args.dump:
            with open(os.path.join(args.dump, item.kname + '.cl'), 'w') as file_:
                src = item.comp.ocl_kernelinfo.src
                file_.write(src)
            with open(os.path.join(args.dump, item.kname + '.tile'), 'w') as file_:
                src = item.comp.hal_compilationinfo.program.code
                file_.write(src)
    print()

    if args.kernel:
        focus = query(by_runtime).single(lambda x: x.kname == args.kernel)
    else:
        focus = by_runtime[-1]

    if args.program:
        print('Program for kernel: {}'.format(focus.kname))
        print(focus.comp.hal_compilationinfo.program.code)
        print()

    print('Source code for kernel: {}'.format(focus.kname))
    print(focus.comp.ocl_kernelinfo.src)
    print()

    print('Memory used by kernel: {}'.format(focus.kname))
    for size, count in focus.comp.hal_compilationinfo.tmp_sizes.items():
        print('  {} tmp(s) of size {}'.format(count, size))
    for size, count in focus.comp.hal_compilationinfo.alloc_sizes.items():
        print('  {} alloc(s) of size {}'.format(count, size))
    print()


if __name__ == '__main__':
    main()
