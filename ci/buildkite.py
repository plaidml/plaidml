#!/usr/bin/env python

import argparse
import glob
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

OUTPUT_ROOT = {
    'Darwin': '/usr/local/var/buildkite-agent/bazel',
    'Linux': '/var/lib/buildkite-agent/bazel',
    'Windows': 'C:\\buildkite-agent\\bazel',
}


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
    if variant == 'macos_x86_64_dbg':
        return ':darwin::sleuth_or_spy:'
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
    if 'llvm-cpu' in pkey:
        return ':crown:'
    if 'opencl-cpu' in pkey:
        return ':crown::cl:'
    else:
        return (':tensorflow:')


def get_shard_emoji(shard):
    numbers = [
        ':zero:',
        ':one:',
        ':two:',
        ':three:',
        ':four:',
        ':five:',
        ':six:',
        ':seven:',
        ':eight:',
        ':nine:',
    ]
    return numbers[shard]


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
        variants.append(dict(
            name=variant,
            python=get_python(variant),
            emoji=get_emoji(variant),
        ))

    tests = []
    for test in util.iterate_tests(plan, args.pipeline):
        if test.shards > 1:
            shard = dict(id=test.shard_id, count=test.shards)
            shard_emoji = get_shard_emoji(test.shard_id)
        else:
            shard = None
            shard_emoji = ''
        tests.append(
            dict(suite=test.suite_name,
                 workload=test.workload_name,
                 platform=test.platform_name,
                 batch_size=test.batch_size,
                 variant=test.variant,
                 timeout=test.timeout,
                 retry=test.retry,
                 soft_fail=test.soft_fail,
                 python=get_python(test.variant),
                 shard=shard,
                 shard_emoji=shard_emoji,
                 emoji=get_emoji(test.variant),
                 engine=get_engine(test.platform_name)))

    if args.count:
        util.printf('variants: {}'.format(len(variants)))
        util.printf('tests   : {}'.format(len(tests)))
        util.printf('total   : {}'.format(len(variants) + len(tests)))
    else:
        ctx = dict(variants=variants, tests=tests)
        yml = pystache.render(load_template('pipeline.yml'), ctx)
        util.printf(yml)


def output_base():
    root = pathlib.Path(OUTPUT_ROOT.get(platform.system()))
    agent = os.getenv('BUILDKITE_AGENT_NAME', 'agent')
    pipeline = os.getenv('BUILDKITE_PIPELINE_SLUG', 'pipeline')
    branch = os.getenv('BUILDKITE_PULL_REQUEST_BASE_BRANCH', '')
    if branch == '':
        branch = os.getenv('BUILDKITE_BRANCH', 'branch')
    return root / agent / pipeline / branch


def cmd_build(args, remainder):
    import yaml
    with open('ci/plan.yml') as file_:
        plan = yaml.safe_load(file_)

    env = os.environ.copy()
    variant = plan['VARIANTS'][args.variant]
    for key, value in variant['env'].items():
        env[key] = str(value)

    explain_log = 'explain.log'
    profile_json = 'profile.json.gz'
    bazel_config = variant.get('bazel_config')
    startup_args = ['--output_base={}'.format(output_base())]

    common_args = []
    if bazel_config:
        common_args += ['--config={}'.format(bazel_config)]
    common_args += ['--define=version={}'.format(args.version)]
    common_args += ['--experimental_generate_json_trace_profile']
    common_args += ['--experimental_json_trace_compression']
    common_args += ['--experimental_profile_cpu_usage']
    common_args += ['--explain={}'.format(explain_log)]
    common_args += ['--profile={}'.format(profile_json)]
    common_args += ['--verbose_failures']
    common_args += ['--verbose_explanations']

    util.printf('--- :bazel: Running Build...')
    if platform.system() == 'Windows':
        util.check_call(['git', 'config', 'core.symlinks', 'true'])
        cenv = util.CondaEnv(pathlib.Path('.cenv'))
        cenv.create('environment-windows.yml')
        env.update(cenv.env())
    util.check_call(['bazelisk'] + startup_args + ['test', '...'] + common_args, env=env)

    util.printf('--- :buildkite: Uploading artifacts...')
    util.buildkite_upload(explain_log)
    util.buildkite_upload(profile_json)

    shutil.rmtree('tmp', ignore_errors=True)
    tarball = os.path.join('bazel-bin', 'pkg.tar.gz')
    with tarfile.open(tarball, "r") as tar:
        wheels = []
        for item in tar.getmembers():
            if item.name.endswith('.whl'):
                wheels.append(item)
        tar.extractall('tmp', members=wheels)
    util.buildkite_upload('*.whl', cwd='tmp')

    variant_dir = os.path.join('tmp', 'build', args.variant)
    os.makedirs(variant_dir)
    shutil.copy(tarball, variant_dir)


def cmd_test(args, remainder):
    import harness
    harness.run(args, remainder)


def make_all_wheels(workdir):
    util.printf('clearing workdir: {}'.format(workdir))
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)

    util.printf('downloading wheels...')
    util.buildkite_download('*.whl', str(workdir), cwd=workdir)

    tarball = 'all_wheels.tar.gz'
    util.printf('creating {}'.format(tarball))
    with tarfile.open(tarball, "w:gz") as tar:
        for whl in workdir.glob('*.whl'):
            util.printf('adding {}'.format(whl))
            tar.add(whl, arcname=whl.name)

    util.printf('uploading {}'.format(tarball))
    util.buildkite_upload(tarball)


def download_test_artifacts(pattern):
    util.buildkite_download(pattern, '.')
    util.buildkite_download(pattern.replace('/', '\\'), '.')
    for path in glob.glob(pattern):
        src = pathlib.Path(path)
        tgt = pathlib.Path(path.replace('\\', '/'))
        tgt.parent.mkdir(parents=True, exist_ok=True)
        src.rename(tgt)


def cmd_report(args, remainder):
    workdir = pathlib.Path('tmp').resolve()
    make_all_wheels(workdir)
    download_test_artifacts('tmp/test/**/*')
    startup_args = ['--output_base={}'.format(output_base())]
    cmd = ['bazelisk'] + startup_args + ['run', '//ci:report']
    cmd += ['--']
    cmd += ['--pipeline', args.pipeline]
    cmd += ['--annotate']
    cmd += [str(workdir)]
    cmd += remainder
    util.check_call(cmd)


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
    parser.add_argument('--shard', type=int)
    parser.add_argument('--shard-count', type=int, default=0)
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
