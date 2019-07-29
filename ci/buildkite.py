#!/usr/bin/env python

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
from distutils.dir_util import copy_tree

# Artifacts are stored with buildkite using the following scheme:
#
# $ROOT/tmp/output/$SUITE/$WORKLOAD/$PLATFORM/{params}/[result.json, result.npy]

if platform.system() == 'Windows':
    ARTIFACTS_ROOT = "\\\\rackstation\\artifacts"
else:
    ARTIFACTS_ROOT = '/nas/artifacts'


def printf(*args, **kwargs):
    excludes_env = {key: kwargs[key] for key in kwargs if key not in ['env']}
    if excludes_env:
        print(*args, excludes_env)
    else:
        print(*args)
    sys.stdout.flush()


def call(cmd, **kwargs):
    printf(cmd, **kwargs)
    subprocess.call(cmd, **kwargs)


def check_call(cmd, **kwargs):
    printf(cmd, **kwargs)
    subprocess.check_call(cmd, **kwargs)


def check_output(cmd, **kwargs):
    printf(cmd, **kwargs)
    return subprocess.check_output(cmd, **kwargs)


def load_template(name):
    this_dir = os.path.dirname(__file__)
    template_path = os.path.join(this_dir, name)
    with open(template_path, 'r') as file_:
        return file_.read()


def first(choices):
    for choice in choices:
        if choice is not None:
            return choice
    return None


class PlanOption(object):

    def __init__(self, suite, workload, platform):
        self._suite = suite
        self._workload = workload
        self._platform = platform

    def get(self, name, default=None):
        """
        precedence order for options:
        - platform_overrides
        - workload
        - suite
        - default
        - None
        """
        override = self._workload.get('platform_overrides', {}).get(self._platform, {}).get(name)
        return first([
            override,
            self._workload.get(name),
            self._suite.get(name),
            default,
        ])


def get_python(variant):
    if variant == 'windows_x86_64':
        return 'python'
    return 'python3'


def buildkite_metadata(key, default=None):
    return os.getenv('BUILDKITE_AGENT_META_DATA_' + key, os.getenv(key, default))


def cmd_pipeline(args, remainder):
    import pystache
    import yaml

    with open('ci/plan.yml') as file_:
        plan = yaml.safe_load(file_)

    variants = []
    for variant in plan['VARIANTS'].keys():
        variants.append(dict(name=variant, python=get_python(variant)))

    tests = []
    for skey, suite in plan['SUITES'].items():
        for pkey, platform in suite['platforms'].items():
            pinfo = plan['PLATFORMS'][pkey]
            variant = pinfo['variant']
            if args.pipeline not in platform['pipelines']:
                continue
            for wkey, workload in suite['workloads'].items():
                popt = PlanOption(suite, workload, pkey)
                skip = workload.get('skip_platforms', [])
                if pkey in skip:
                    continue
                for batch_size in suite['params'][args.pipeline]['batch_sizes']:
                    tests.append(
                        dict(
                            suite=skey,
                            workload=wkey,
                            platform=pkey,
                            batch_size=batch_size,
                            variant=variant,
                            timeout=popt.get('timeout', 20),
                            retry=popt.get('retry'),
                            python=get_python(variant),
                        ))

    if args.count:
        print('variants: {}'.format(len(variants)))
        print('tests   : {}'.format(len(tests)))
        print('total   : {}'.format(len(variants) + len(tests)))
    else:
        ctx = dict(variants=variants, tests=tests)
        yml = pystache.render(load_template('pipeline.yml'), ctx)
        print(yml)


def cmd_build(args, remainder):
    common_args = []
    common_args += ['--config={}'.format(args.variant)]
    common_args += ['--define=version={}'.format(args.version)]
    common_args += ['--verbose_failures']
    if platform.system() == 'Windows':
        # TODO: Test everything on windows
        check_call(['git', 'config', 'core.symlinks', 'true'])
        check_call(['bazelisk', 'build', ':pkg'] + common_args)
    else:
        check_call(['bazelisk', 'test', '...'] + common_args)
    archive_dir = os.path.join(ARTIFACTS_ROOT, args.pipeline, args.build_id, 'build', args.variant)
    os.makedirs(archive_dir, exist_ok=True)
    shutil.copy(os.path.join('bazel-bin', 'pkg.tar.gz'), archive_dir)


def cmd_test(args, remainder):
    print('cmd_test')
    # import yaml

    # root = pathlib.Path('.').resolve() / 'tmp'
    # input = root / 'input'
    # output = root / 'output' / args.suite / args.workload / args.platform / 'BATCH_SIZE={}'.format(
    #     args.batch_size)

    # with open('ci/plan.yml') as fp:
    #     plan = yaml.safe_load(fp)

    # platform = plan['PLATFORMS'][args.platform]
    # variant_name = platform['variant']
    # variant = plan['VARIANTS'][variant_name]
    # arch = variant['arch']

    # suites = plan['SUITES']
    # suite = suites.get(args.suite)
    # if suite is None:
    #     sys.exit('Invalid suite. Available suites: {}'.format(list(suites)))
    # platform_cfg = suite['platforms'][args.platform]

    # workload = suite['workloads'].get(args.workload)
    # if workload is None:
    #     sys.exit('Invalid workload. Available workloads: {}'.format(list(suite['workloads'])))

    # popt = PlanOption(suite, workload, args.platform)

    # shutil.rmtree(input, ignore_errors=True)
    # archive_dir = pathlib.Path(ARTIFACTS_ROOT) / args.pipeline / args.build_id
    # if args.local:
    #     pkg_path = pathlib.Path('bazel-bin/pkg.tar.gz')
    #     outdir = root / 'nas'
    #     version = '0.0.0.dev0'
    # else:
    #     pkg_path = archive_dir / 'build' / variant_name / 'pkg.tar.gz'
    #     outdir = archive_dir
    #     version = args.version
    # with tarfile.open(pkg_path, 'r') as tar:
    #     tar.extractall(input)

    # shutil.rmtree(output, ignore_errors=True)
    # output.mkdir(parents=True)

    # cwd = input / popt.get('cwd', '.')
    # spec = input / popt.get('conda_env')

    # printf('--- Creating conda env from {}'.format(spec))
    # instance_name = os.getenv('BUILDKITE_AGENT_NAME', 'harness')
    # sig = hashlib.md5()
    # sig.update(spec.read_bytes())
    # base_path = pathlib.Path('~', '.t2', instance_name, sig.hexdigest()).expanduser()

    # base_env = util.CondaEnv(base_path)
    # base_env.create(spec)
    # conda_env = base_env.clone(root / pathlib.Path('cenv'))
    # env = os.environ.copy()
    # env.update(conda_env.env())

    # for whl in popt.get('wheels', []):
    #     whl_filename = whl.format(arch=arch, version=version)
    #     whl_path = input / whl_filename
    #     conda_env.install(whl_path)

    # if 'cuda' in args.platform:
    #     env['CUDA_VISIBLE_DEVICES'] = buildkite_metadata('CUDA_VISIBLE_DEVICES', '0')

    # if 'stripe' in args.platform:
    #     env['USE_STRIPE'] = '1'

    # env['PLAIDML_DEVICE_IDS'] = buildkite_metadata('PLAIDML_DEVICE_IDS')
    # env['PLAIDML_EXPERIMENTAL'] = buildkite_metadata('PLAIDML_EXPERIMENTAL', '0')

    # with (output / 'env.json').open('w') as fp:
    #     printf('Writing:', fp.name)
    #     json.dump(env, fp)

    # printf('--- Running test {suite}/{workload} on {platform}'.format(
    #     suite=args.suite,
    #     workload=args.workload,
    #     platform=args.platform,
    # ))

    # cmd_args = platform_cfg.get('prepend_args', []) + popt.get('prepend_args', [])
    # cmd_args += platform_cfg.get('args', []) + popt.get('args', [])
    # cmd_args += platform_cfg.get('append_args', []) + popt.get('append_args', [])
    # ctx = dict(
    #     results=output,
    #     batch_size=args.batch_size,
    #     workload=args.workload,
    # )
    # cmd_args = [str(x).format(**ctx) for x in cmd_args]
    # if 'stripe' in args.platform:
    #     try:
    #         cmd_args.remove('--no-kernel-timing')
    #     except ValueError:
    #         pass

    # cmd = [popt.get('runner')] + cmd_args
    # check_call(cmd, cwd=cwd, env=env)

    # src = root / 'output'
    # dst = outdir / 'test'
    # copy_tree(str(src), str(dst))


def cmd_analysis(args, remainder):
    archive_dir = os.path.join(ARTIFACTS_ROOT, args.pipeline, args.build_id)
    cmd = ['bazelisk', 'run', '//ci:analysis']
    cmd += ['--']
    cmd += ['--pipeline', args.pipeline]
    cmd += ['--annotate']
    cmd += [archive_dir]
    cmd += remainder
    check_call(cmd)


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


def make_cmd_analysis(parent):
    parser = parent.add_parser('analysis')
    parser.set_defaults(func=cmd_analysis)


def make_cmd_pipeline(parent):
    parser = parent.add_parser('pipeline')
    parser.add_argument('--count', action='store_true')
    parser.set_defaults(func=cmd_pipeline)


def main():

    pipeline = os.getenv('PIPELINE', 'pr')
    branch = os.getenv('BUILDKITE_BRANCH', 'undefined')
    build_id = os.getenv('BUILDKITE_BUILD_NUMBER', '0')
    with open('VERSION', 'r') as verf:
        version = verf.readline().strip()
    default_version = os.getenv('VAI_VERSION', '{}+{}.dev{}'.format(version, pipeline, build_id))

    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--pipeline', default=pipeline)
    main_parser.add_argument('--branch', default=branch)
    main_parser.add_argument('--build_id', default=build_id)
    main_parser.add_argument('--version', default=default_version)

    sub_parsers = main_parser.add_subparsers()

    make_cmd_pipeline(sub_parsers)
    make_cmd_build(sub_parsers)
    make_cmd_test(sub_parsers)
    make_cmd_analysis(sub_parsers)

    args, remainder = main_parser.parse_known_args()
    if 'func' not in args:
        main_parser.print_help()
        return

    path = os.getenv('PATH').split(os.pathsep)
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        path.insert(0, '/usr/local/miniconda3/bin')
    os.environ.update({'PATH': os.pathsep.join(path)})

    args.func(args, remainder)


if __name__ == '__main__':
    main()
