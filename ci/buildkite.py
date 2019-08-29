#!/usr/bin/env python

import argparse
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile

import util

# Artifacts are stored with buildkite using the following scheme:
#
# $ROOT/tmp/output/$SUITE/$WORKLOAD/$PLATFORM/{params}/[result.json, result.npy]

if platform.system() == 'Windows':
    ARTIFACTS_ROOT = "\\\\rackstation\\artifacts"
else:
    ARTIFACTS_ROOT = '/nas/artifacts'


def load_template(name):
    this_dir = os.path.dirname(__file__)
    template_path = os.path.join(this_dir, name)
    with open(template_path, 'r') as file_:
        return file_.read()


def get_emoji(variant):
    if variant == 'windows_x86_64':
        return ':windows:'
    if variant == 'macos_x86_64':
        return ':darwin:'
    return ':linux:'


def get_engine(pkey):
    if 'stripe-ocl' in pkey:
        return ':barber::cl:'
    if 'stripe-mtl' in pkey:
        return ':barber::metal:'
    if 'plaid-mtl' in pkey:
        return ':black_square_button::metal:'
    if 'plaid-ocl' in pkey:
        return ':black_square_button::cl:'
    else:
        return (':tensorflow:')


def get_shard_number(shard):
    numbers = [
        ':zero:', ':one:', ':two:', ':three:', ':four:', ':five:', ':six:', ':seven:', ':eight:',
        ':nine:'
    ]
    return (numbers[shard])


def get_python(variant):
    if variant == 'windows_x86_64':
        return 'python'
    return 'python3'


def cmd_pipeline(args, remainder):
    import pystache
    import yaml

    with open('ci/plan.yml') as file_:
        plan = yaml.safe_load(file_)

    variants = []
    for variant in plan['VARIANTS'].keys():
        variants.append(dict(name=variant, python=get_python(variant), emoji=get_emoji(variant)))

    tests = []
    for skey, suite in plan['SUITES'].items():
        for pkey, platform in suite['platforms'].items():
            pinfo = plan['PLATFORMS'][pkey]
            variant = pinfo['variant']
            if args.pipeline not in platform['pipelines']:
                continue

            for wkey, workload in suite['workloads'].items():
                popt = util.PlanOption(suite, workload, pkey)
                skip = workload.get('skip_platforms', [])
                if pkey in skip:
                    continue

                shc = popt.get('shardcount')

                if shc:
                    for shard in popt.get('run_shards'):
                        rsh = shard
                        shard = 'shard'
                        shard_emoji = get_shard_number(rsh)
                        for batch_size in suite['params'][args.pipeline]['batch_sizes']:
                            tests.append(
                                dict(suite=skey,
                                     workload=wkey,
                                     platform=pkey,
                                     batch_size=batch_size,
                                     variant=variant,
                                     timeout=popt.get('timeout', 20),
                                     retry=popt.get('retry'),
                                     softfail=popt.get('softfail'),
                                     python=get_python(variant),
                                     shard=shard,
                                     shardcount=shc,
                                     rsh=rsh,
                                     shard_emoji=shard_emoji,
                                     emoji=get_emoji(variant),
                                     engine=get_engine(pkey)))

                else:
                    shard = None

                for batch_size in suite['params'][args.pipeline]['batch_sizes']:
                    if shc:
                        continue

                    else:
                        tests.append(
                            dict(suite=skey,
                                 workload=wkey,
                                 platform=pkey,
                                 batch_size=batch_size,
                                 variant=variant,
                                 timeout=popt.get('timeout', 20),
                                 retry=popt.get('retry'),
                                 softfail=popt.get('softfail'),
                                 python=get_python(variant),
                                 emoji=get_emoji(variant),
                                 engine=get_engine(pkey)))

    if args.count:
        print('variants: {}'.format(len(variants)))
        print('tests   : {}'.format(len(tests)))
        print('total   : {}'.format(len(variants) + len(tests)))
    else:
        ctx = dict(variants=variants, tests=tests)
        yml = pystache.render(load_template('pipeline.yml'), ctx)
        print(yml)


def bazel_bin_dir():
    out = util.check_output(['bazelisk', 'info', 'bazel-bin'], stderr=subprocess.DEVNULL)
    return out.decode().strip('\n')


def wheel_path(arg):
    return pathlib.Path(bazel_bin_dir()) / arg / 'wheel.pkg' / 'dist'


def wheel_clean(arg):
    ws = list(wheel_path(arg).glob('*.whl'))
    for f in ws:
        if f.is_file():
            print('deleting: ' + str(f.resolve()))
            try:
                os.remove(f)
            except PermissionError:
                import stat
                os.chmod(f, stat.S_IWRITE)
                os.remove(f)


def cmd_build(args, remainder):

    util.printf('--- :snake: pre-build steps... ')
    util.printf('bazel shutdown...')
    util.check_output(['bazelisk', 'shutdown'])
    util.printf('delete any old whl files...')
    wheel_clean('plaidml')
    wheel_clean('plaidbench')
    wheel_clean('plaidml/keras')

    common_args = []
    common_args += ['--config={}'.format(args.variant)]
    common_args += ['--define=version={}'.format(args.version)]
    common_args += ['--explain={}'.format(os.path.abspath('explain.log'))]
    common_args += ['--verbose_failures']
    common_args += ['--verbose_explanations']

    util.printf('--- :bazel: Running Build ...')

    if platform.system() == 'Windows':
        util.check_call(['git', 'config', 'core.symlinks', 'true'])
        cenv = util.CondaEnv(pathlib.Path('.cenv'))
        cenv.create('environment-windows.yml')
        env = os.environ.copy()
        env.update(cenv.env())
    else:
        env = None

    util.check_call(['bazelisk', 'test', '...'] + common_args, env=env)
    archive_dir = os.path.join(
        args.root,
        args.pipeline,
        args.build_id,
        'build',
        args.variant,
    )

    util.printf('--- :buildkite: Uploading artifacts...')
    pw = wheel_path('plaidml').resolve()
    pbw = wheel_path('plaidbench').resolve()
    pkw = wheel_path('plaidml/keras').resolve()
    util.check_call(['buildkite-agent', 'artifact', 'upload', '*.whl'], cwd=pw)
    util.check_call(['buildkite-agent', 'artifact', 'upload', '*.whl'], cwd=pbw)
    util.check_call(['buildkite-agent', 'artifact', 'upload', '*.whl'], cwd=pkw)
    util.check_call(['buildkite-agent', 'artifact', 'upload', 'explain.log'])
    os.makedirs(archive_dir, exist_ok=True)
    shutil.copy(os.path.join('bazel-bin', 'pkg.tar.gz'), archive_dir)
    util.printf('bazel shutdown...')
    util.check_output(['bazelisk', 'shutdown'])


def cmd_test(args, remainder):
    import harness
    harness.run(args, remainder)


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def cmd_pack(arg):
    pathlib.Path(arg).mkdir(parents=True, exist_ok=True)
    dir_list = ['windows_x86_64', 'macos_x86_64', 'linux_x86_64', 'common']
    whl_type = ['win_amd64', 'macosx', 'manylinux', 'none-any']

    os.chdir(arg)
    print('downloading wheels...')
    util.check_output(['buildkite-agent', 'artifact', 'download', '*.whl', '.'], cwd='.')
    whl_files = list(pathlib.Path('.').glob('*.whl'))

    [pathlib.Path(d).mkdir(parents=True, exist_ok=True) for d in dir_list]

    for whl, dest in zip(whl_type, dir_list):
        for f in whl_files:
            if whl in str(f):
                shutil.move(str(f), dest)

    print('packing all_wheels...')
    make_tarfile('all_wheels.tar.gz', '.')
    util.check_call(['buildkite-agent', 'artifact', 'upload', 'all_wheels.tar.gz'])
    os.chdir('..')


def cmd_report(args, remainder):
    cmd_pack('tmp')
    #shutil.rmtree('tmp')

    archive_dir = os.path.join(args.root, args.pipeline, args.build_id)
    cmd = ['bazelisk', 'run', '//ci:report']
    cmd += ['--']
    cmd += ['--pipeline', args.pipeline]
    cmd += ['--annotate']
    cmd += [archive_dir]
    cmd += remainder
    util.check_call(cmd, stderr=subprocess.DEVNULL)


def make_cmd_build(parent):
    parser = parent.add_parser('build')
    parser.add_argument('variant')
    parser.set_defaults(func=cmd_build)


def make_cmd_test(parent):
    parser = parent.add_parser('test')
    parser.add_argument('platform')
    parser.add_argument('suite')
    parser.add_argument('workload')
    parser.add_argument('batch_size')
    parser.add_argument('--local', action='store_true')
    parser.set_defaults(func=cmd_test)


def make_cmd_report(parent):
    parser = parent.add_parser('report')
    parser.set_defaults(func=cmd_report)


def make_cmd_pipeline(parent):
    parser = parent.add_parser('pipeline')
    parser.add_argument('--count', action='store_true')
    parser.set_defaults(func=cmd_pipeline)


def main():
    pipeline = os.getenv('PIPELINE', 'plaidml')
    branch = os.getenv('BUILDKITE_BRANCH', 'undefined')
    build_id = os.getenv('BUILDKITE_BUILD_NUMBER', '0')
    with open('VERSION', 'r') as verf:
        version = verf.readline().strip()
    default_version = os.getenv('VAI_VERSION', '{}+{}.dev{}'.format(version, pipeline, build_id))

    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--root', default=ARTIFACTS_ROOT)
    main_parser.add_argument('--pipeline', default=pipeline)
    main_parser.add_argument('--branch', default=branch)
    main_parser.add_argument('--build_id', default=build_id)
    main_parser.add_argument('--version', default=default_version)

    sub_parsers = main_parser.add_subparsers()

    make_cmd_pipeline(sub_parsers)
    make_cmd_build(sub_parsers)
    make_cmd_test(sub_parsers)
    make_cmd_report(sub_parsers)

    args, remainder = main_parser.parse_known_args()
    if 'func' not in args:
        main_parser.print_help()
        return

    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        path = os.getenv('PATH').split(os.pathsep)
        path.insert(0, '/usr/local/miniconda3/bin')
        os.environ.update({'PATH': os.pathsep.join(path)})

    args.func(args, remainder)


if __name__ == '__main__':
    main()
