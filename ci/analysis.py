#!/usr/bin/env python3

import argparse
import base64
import io
import json
import mimetypes
import os
import subprocess
import sys
import tarfile

import matplotlib
matplotlib.use('Agg')  # this must come before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import pystache
import yaml
from asq.initiators import query
from asq.record import new

DEFAULT_RATIO_THRESHOLD = 0.8
DEFAULT_BUILD_URL = 'https://buildkite.com/vertex-dot-ai'
DEFAULT_CONVERT_TIMEOUT = 480
PLAN_PATH = os.path.abspath('ci/plan.yml')
GOLDEN_ROOT = os.path.abspath('ci/golden')


def printf(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def check_call(cmd, **kwargs):
    printf(cmd)
    subprocess.check_call(cmd, **kwargs)


def first(choices):
    for choice in choices:
        if choice is not None:
            return choice
    return None


def get_option(name, suite, workload, default=None):
    """
    precedence order for options:
      - workload
      - suite
      - default
      - None
    """
    return first([
        workload.get(name),
        suite.get(name),
        default,
    ])


class Platform(object):

    def __init__(self, text, gpu_flops):
        parts = text.split('-')
        self.full = text
        self.framework = parts[0]
        self.runtime = '-'.join(parts[1:1])
        self.gpu = parts[3]
        self.gpu_flops = gpu_flops.get(self.gpu)

    def __repr__(self):
        return '<Platform({})>'.format(self.full)


class Iterator(object):

    def __init__(self, suite, workload, platform, batch_size):
        self.suite_name, self.suite = suite
        self.workload_name, self.workload = workload
        self.platform_name, self.platform = platform
        self.batch_size = batch_size

    def __repr__(self):
        return '{}/{}/{}/bs{}'.format(self.suite_name, self.workload_name, self.platform_name,
                                      self.batch_size)


class RawResult(object):

    def __init__(self, root, it):
        self.it = it

        self.path = os.path.join(root, it.suite_name, it.workload_name, it.platform_name,
                                 'BATCH_SIZE={}'.format(it.batch_size))

        data = {}
        result_path = os.path.join(self.path, 'result.json')
        if os.path.exists(result_path):
            try:
                with open(result_path) as file_:
                    data = json.load(file_)
            except Exception as ex:
                data = {'exception': str(ex)}

        self.exception = data.get('exception')
        self.compile_duration = data.get('compile_duration')
        self.execution_duration = data.get('duration_per_example', data.get('execution_duration'))

        self.np_data = None
        np_result_path = os.path.join(self.path, 'result.npy')
        if os.path.exists(np_result_path):
            try:
                self.np_data = np.load(np_result_path)
            except Exception as ex:
                printf('  Exception:', ex)
                if self.exception is None:
                    self.exception = str(ex)

        self.env = {}
        env_path = os.path.join(self.path, 'env.json')
        if os.path.exists(env_path):
            with open(env_path) as file_:
                self.env = json.load(file_)
            self.build_url = '{}#{}'.format(self.env.get('BUILDKITE_BUILD_URL'),
                                            self.env.get('BUILDKITE_JOB_ID'))
        else:
            self.build_url = os.getenv('BUILDKITE_BUILD_URL', DEFAULT_BUILD_URL)

        self.eventlog_path = os.path.join(self.path, 'eventlog.gz')
        self.profile_path = os.path.join(self.path, 'profile.html')

    def convert_eventlog(self):
        if os.path.exists(self.eventlog_path):
            printf('  Converting eventlog...')
            notebook = os.path.abspath(os.path.join('tools', 'analysis', 'Analysis.ipynb'))
            timeout = '--ExecutePreprocessor.timeout={}'.format(DEFAULT_CONVERT_TIMEOUT)
            env = os.environ.copy()
            env['PLAIDML_EVENTLOG_FILENAME'] = self.eventlog_path
            # This crazy way of calling nbconvert is because bash shebang lines only support
            # a max of 128 chars. It's very likely that in this situation, the shebang line
            # that points to python will be > 128 chars.
            nbconvert = os.path.join(os.getenv('CONDA_DEFAULT_ENV'), 'bin', 'jupyter-nbconvert')
            cmd = [
                'python', nbconvert, '--execute', '--output', self.profile_path, timeout, notebook
            ]
            check_call(cmd, env=env)

    def exists(self):
        return os.path.exists(self.path)


class TestResult(object):

    def __init__(self, skip, compare):
        self.errors = []
        self.failures = []
        self.skips = []
        self.expected = None
        self.skipped = skip
        self.compare = compare

    def add_error(self, msg):
        printf('  ERROR:', msg)
        self.errors.append(msg)

    def add_failure(self, msg):
        printf('  FAIL:', msg)
        self.failures.append(msg)

    def add_skip(self, msg):
        printf('  SKIP:', msg)
        self.skips.append(msg)

    def set_expected(self, msg):
        printf('  SKIP:', msg)
        self.expected = msg

    def status(self):
        if self.errors:
            if self.skipped:
                return 'SKIP'
            return 'ERROR'
        elif self.failures:
            if self.skipped:
                return 'SKIP'
            return 'FAIL'
        elif self.expected:
            return 'SKIP'
        elif len(self.skips) > 0:
            return 'SKIP'
        return 'PASS'

    def reason(self):
        parts = []
        if self.errors:
            parts += map(lambda x: 'ERROR: ' + x, self.errors)
        if self.failures:
            parts += map(lambda x: 'FAIL: ' + x, self.failures)
        if self.skips:
            parts += map(lambda x: 'SKIP: ' + x, self.skips)
        if self.expected:
            parts += ['SKIP: ' + self.expected]
        return '\n'.join(parts)


class Result(object):

    def __init__(self, root, it, golden_it):
        self.it = it
        # The current results
        self.cur = RawResult(root, it)
        # The last results matching the platform of the current results
        self.ref = RawResult(GOLDEN_ROOT, it)
        # The golden results for the baseline platform (usually a TF/CUDA variant)
        self.golden = RawResult(GOLDEN_ROOT, golden_it)

        self.ratio = None
        if self.cur.execution_duration and self.ref.execution_duration:
            self.ratio = self.ref.execution_duration / self.cur.execution_duration

        cur_efficiency = None
        base_efficiency = None
        if self.cur.execution_duration and self.it.platform.gpu_flops:
            cur_efficiency = self.cur.execution_duration * self.it.platform.gpu_flops
        if self.golden.execution_duration:
            base_efficiency = self.golden.execution_duration * golden_it.platform.gpu_flops

        self.efficiency = None
        if cur_efficiency and base_efficiency:
            self.efficiency = base_efficiency / cur_efficiency

        label_parts = [self.it.platform.gpu, self.it.workload_name]
        if self.it.batch_size:
            label_parts += [str(self.it.batch_size)]
        self.label = '-'.join(label_parts)

        self.test_result = self._check_result()

        try:
            self.cur.convert_eventlog()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            self.test_result.add_error(str(ex))

    def __repr__(self):
        return '<Result({})>'.format(self.it)

    def _check_result(self):
        printf(self.it, self.cur.compile_duration, self.ref.execution_duration,
               self.cur.execution_duration, self.ratio, self.efficiency)

        skip = self.it.workload.get('skip', False)
        expected = self.it.workload.get('expected')
        precision = self.it.workload.get('precision')
        perf_threshold = self.it.workload.get('perf_threshold', DEFAULT_RATIO_THRESHOLD)
        correct = self.it.workload.get('correct', True)
        compare = get_option('compare', self.it.suite, self.it.workload, True)

        if not self.cur.exists():
            printf('  missing cur')
        if not compare and not self.ref.exists():
            printf('  missing ref')

        test_result = TestResult(skip, compare)

        try:
            if self.cur.exception:
                first_line = self.cur.exception.split('\n')[0]
                if expected:
                    if expected not in self.cur.exception:
                        test_result.add_failure('Expected: %r' % expected)
                    else:
                        test_result.set_expected(first_line)
                else:
                    test_result.add_failure(first_line)
            elif compare:
                if not self.ref.execution_duration:
                    test_result.add_error('Missing reference duration')
                elif not self.cur.execution_duration:
                    test_result.add_error('Missing result duration')
                else:
                    if self.ratio < perf_threshold:
                        test_result.add_failure('Performance regression')

                base_output = self.golden.np_data
                if precision != 'untested':
                    # If base_output is None and precision == 'untested' then
                    # this is interpreted to mean no correctness test is desired;
                    # so no error that it's missing in result.
                    if base_output is None:
                        test_result.add_error('Golden correctness data not found')
                    else:
                        if self.cur.np_data is None:
                            test_result.add_error('Missing correctness test output')
                        else:
                            self._check_correctness(base_output, self.cur.np_data, test_result,
                                                    precision, correct)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            test_result.add_error(str(ex))
        return test_result

    def _check_correctness(self, base_output, cur_output, test_result, precision,
                           should_be_correct):

        # TODO: Parameterize relative and absolute error tolerance
        if precision == 'high':
            rel_err = 1e-04
        elif precision == 'low':
            rel_err = 0.2
        else:
            test_result.add_error('Unexpected precision {!r} in test suite'.format(precision))

        correct = np.allclose(base_output, cur_output, rtol=rel_err, atol=2e-05)
        # This duplicates allclose calculation for more detailed report
        relative_error = ((rel_err * np.absolute(base_output - cur_output)) /
                          (1e-06 + rel_err * np.absolute(cur_output)))
        max_error = np.amax(relative_error)
        max_abs_error = np.amax(np.absolute(base_output - cur_output))
        correct_entries = 0
        incorrect_entries = 0
        for x in np.nditer(relative_error):
            if x > rel_err:
                incorrect_entries += 1
            else:
                correct_entries += 1
        try:
            fail_ratio = incorrect_entries / float(correct_entries + incorrect_entries)
        except ZeroDivisionError:
            fail_ratio = 'Undefined'

        if not correct:
            if should_be_correct:
                msg = 'Correctness failure: {}, max_abs_error: {}, fail rate: {}'
                test_result.add_failure(msg.format(max_error, max_abs_error, fail_ratio))
            else:
                msg = 'Correctness failure (expected): {}, max_abs_error: {}, fail rate: {}'
                test_result.add_skip(msg.format(max_error, max_abs_error, fail_ratio))


def collect_results(root, pipeline):
    with open(PLAN_PATH) as file_:
        plan = yaml.safe_load(file_)
    gpu_flops = plan['CONST']['gpu_flops']
    baseline_name = plan['CONST']['efficiency_baseline']
    for suite_name, suite in plan['SUITES'].items():
        for workload_name, workload in suite['workloads'].items():
            skip_platforms = workload.get('skip_platforms', [])
            for platform_name, platform in suite['platforms'].items():
                if platform_name in skip_platforms or platform_name == baseline_name:
                    continue
                if pipeline not in platform['pipelines']:
                    continue
                for batch_size in suite['params'][pipeline]['batch_sizes']:
                    it = Iterator(
                        (suite_name, suite),
                        (workload_name, workload),
                        (platform_name, Platform(platform_name, gpu_flops)),
                        batch_size,
                    )
                    baseline_it = Iterator(
                        (suite_name, suite),
                        (workload_name, workload),
                        (baseline_name, Platform(baseline_name, gpu_flops)),
                        batch_size,
                    )
                    yield Result(root, it, baseline_it)


CSS_MAP = {
    'ERROR': 'background-color: red; color: white',
    'FAIL': 'background-color: red; color: white',
    'SKIP': 'background-color: yellow',
    'PASS': 'background-color: green; color: white',
}


def load_template(name):
    this_dir = os.path.dirname(__file__)
    template_path = os.path.join(this_dir, 'templates', name)
    with open(template_path, 'r') as file_:
        return file_.read()


def ratio_plot(path, labels, values, title):
    y_pos = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots()
    fig.set_size_inches(6, len(values) / 4)

    # Vertical lines with major at 1.0
    ax.xaxis.grid(True, color='666666')
    ax.axvline(1.0, color='gray')

    # Horizontal bar chart, labeled on y axis with test config
    plt.title(title)
    plt.barh(y_pos, values, zorder=10)
    plt.yticks(y_pos, labels)

    for patch, value in zip(ax.patches, values):
        ax.text(patch.get_width() - 0.06,
                patch.get_y() + 0.01,
                '%.2f' % round(value, 2),
                ha='center',
                va='bottom',
                color='white',
                zorder=20)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def generate_ratio_chart(results, report_dir):
    values = query(results) \
        .where(lambda x: x.test_result.compare) \
        .where(lambda x: x.ratio) \
        .order_by(lambda x: x.label) \
        .select(lambda x: x.ratio) \
        .to_list()
    labels = query(results) \
        .where(lambda x: x.test_result.compare) \
        .where(lambda x: x.ratio) \
        .order_by(lambda x: x.label) \
        .select(lambda x: x.label) \
        .to_list()
    if len(values):
        filename = os.path.join(report_dir, 'ratios.png')
        ratio_plot(filename, labels, values, 'Throughput compared to golden')
        return Image(filename)
    return None


def generate_efficiency_chart(results, report_dir):
    values = query(results) \
        .where(lambda x: x.test_result.compare) \
        .where(lambda x: x.efficiency) \
        .order_by(lambda x: x.label) \
        .select(lambda x: x.efficiency) \
        .to_list()
    labels = query(results) \
        .where(lambda x: x.test_result.compare) \
        .where(lambda x: x.efficiency) \
        .order_by(lambda x: x.label) \
        .select(lambda x: x.label) \
        .to_list()
    if len(values):
        filename = os.path.join(report_dir, 'efficiency.png')
        ratio_plot(filename, labels, values, 'Efficiency relative to TF/GP100')
        return Image(filename)
    return None


def render_float(value):
    if value:
        return '{0:.3f}'.format(value)
    return 'N/A'


def make_html_results(results):

    def _make_one_result(x):
        return new(
            label=x.label,
            status_css=CSS_MAP.get(x.test_result.status()),
            status=x.test_result.status(),
            gpu=x.it.platform.gpu,
            workload=x.it.workload_name,
            batch_size=x.it.batch_size,
            cur_com=render_float(x.cur.compile_duration),
            cur_run=render_float(x.cur.execution_duration),
            ref_run=render_float(x.ref.execution_duration),
            ratio=render_float(x.ratio),
            log=x.cur.build_url,
            reason=x.test_result.reason(),
        )

    return query(results) \
        .select(_make_one_result) \
        .order_by(lambda x: x.label) \
        .to_list()


def make_html_suites(results):
    return query(results) \
        .group_by(
            lambda x: x.it.suite_name,
            result_selector=lambda k, g: new(name=k, results=make_html_results(g))) \
        .order_by(lambda x: x.name) \
        .to_list()


def make_html_summary(results):
    counts = query(results) \
        .group_by(lambda x: x.test_result.status()) \
        .to_dictionary(lambda x: x.key, len)

    errors = counts.get('ERROR', 0)
    failures = counts.get('FAIL', 0)
    skipped = counts.get('SKIP', 0)
    passed = counts.get('PASS', 0)
    total = errors + failures + skipped + passed

    if errors:
        status = 'ERROR'
    elif failures:
        status = 'FAIL'
    else:
        status = 'PASS'

    return new(
        status=status,
        css=CSS_MAP.get(status),
        errors_count=errors,
        failures_count=failures,
        skipped_count=skipped,
        passed_count=passed,
        total_count=total,
    )


def make_html_failures(results, status):
    failures = query(results) \
        .where(lambda x: x.test_result.status() == status) \
        .select(lambda x: new(
            name=x.label,
            body=x.test_result.reason(),
            job_url=x.cur.build_url)) \
        .order_by(lambda x: x.name) \
        .to_list()
    if len(failures):
        return {'count': len(failures), 'items': failures}
    return None


def is_skipped(record):
    return record.test_result.status() == 'SKIP'


def make_junit_failure(record):
    if record.test_result.status() == 'FAIL':
        msg = '; '.join(record.test_result.failures)
        return new(message=msg)
    return None


def make_junit_error(record):
    if record.test_result.status() == 'ERROR':
        msg = '; '.join(record.test_result.errors)
        return new(message=msg)
    return None


def make_junit_stdout(record):
    reason = record.test_result.reason()
    if reason:
        return new(text=reason)
    return None


def make_junit_context(results):
    testcases = query(results) \
        .select(lambda x: new(
            classname=x.it.platform.full,
            name='{}-{}'.format(x.it.workload_name, x.it.batch_size),
            time=x.cur.execution_duration,
            skipped=is_skipped(x),
            failure=make_junit_failure(x),
            error=make_junit_error(x),
            stdout=make_junit_stdout(x))) \
        .to_list()
    return dict(testcases=testcases)


class Image(object):

    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)

    def artifact_url(self):
        return 'artifact://analysis/{}'.format(self.filename)

    def data_url(self):
        mime, _ = mimetypes.guess_type(self.path)
        with open(self.path, 'rb') as fp:
            data = fp.read()
        data64 = base64.b64encode(data).decode()
        return 'data:{};base64,{}'.format(mime, data64)


def write_file(filename, content):
    printf('Writing:', filename)
    with open(filename, 'w') as file_:
        file_.write(content)


def buildkite_annotate(root, style, html):
    printf('--- Uploading artifacts and adding annotations')
    check_call(['buildkite-agent', 'artifact', 'upload', 'analysis/**/*'], cwd=root)

    cmd = ['buildkite-agent', 'annotate', '--style', style]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(html.encode())


def main():
    printf('--- Analyzing test results')
    printf('PATH:', os.getenv('PATH'))

    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--pipeline', default='pr')
    parser.add_argument('--annotate', action='store_true')
    args = parser.parse_args()

    test_dir = os.path.join(args.root, 'test')
    analysis_dir = os.path.join(args.root, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    results = list(collect_results(test_dir, args.pipeline))

    ratio_png = generate_ratio_chart(results, analysis_dir)
    efficiency_png = generate_efficiency_chart(results, analysis_dir)

    xml = pystache.render(load_template('junit.xml'), make_junit_context(results))
    write_file(os.path.join(analysis_dir, 'junit.xml'), xml)

    summary = make_html_summary(results)
    context = {
        'suites': make_html_suites(results),
        'summary': summary,
    }
    if ratio_png:
        context['ratio_png'] = ratio_png.data_url()
    if efficiency_png:
        context['efficiency_png'] = efficiency_png.data_url()
    html = pystache.render(load_template('report.html'), context)
    write_file(os.path.join(analysis_dir, 'report.html'), html)

    if summary.status == 'PASS':
        style = 'success'
    else:
        style = 'error'
    write_file(os.path.join(analysis_dir, 'status.txt'), style)

    context = {
        'summary': summary,
        'errors': make_html_failures(results, 'ERROR'),
        'failures': make_html_failures(results, 'FAIL'),
        'report_url': 'artifact://analysis/report.html',
    }
    html = pystache.render(load_template('annotate.html'), context)
    write_file(os.path.join(analysis_dir, 'annotate.html'), html)

    if args.annotate:
        buildkite_annotate(args.root, style, html)

    if summary.status != 'PASS':
        sys.exit(1)


if __name__ == '__main__':
    main()
