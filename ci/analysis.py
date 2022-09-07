#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

GOLDEN_ROOT = Path('ci/golden')
DEFAULT_BUILD_URL = 'https://buildkite.com/plaidml'
DEFAULT_PERF_THRESHOLD = 0.7


class RawResult:

    def __init__(self, path):
        self.path = path

        data = {}
        result_path = self.path / 'result.json'
        if result_path.exists():
            try:
                with result_path.open() as fp:
                    data = json.load(fp)
            except Exception as ex:
                data = {'exception': str(ex)}

        self.exception = data.get('exception')
        self.compile_duration = data.get('compile_duration')
        self.execution_duration = data.get('duration_per_example', data.get('execution_duration'))

        self.np_data = None
        np_result_path = self.path / 'result.npy'
        if np_result_path.exists():
            try:
                self.np_data = np.load(np_result_path)
            except Exception as ex:
                print('  Exception:', ex)
                if self.exception is None:
                    self.exception = str(ex)

    def exists(self):
        return self.path.exists()


class TestResult:

    def __init__(self, skip, compare):
        self.errors = []
        self.failures = []
        self.skips = []
        self.expected = None
        self.skipped = skip
        self.compare = compare

    def add_error(self, msg):
        print('  ERROR:', msg)
        self.errors.append(msg)

    def add_failure(self, msg):
        print('  FAIL:', msg)
        self.failures.append(msg)

    def add_skip(self, msg):
        print('  SKIP:', msg)
        self.skips.append(msg)

    def set_expected(self, msg):
        print('  SKIP:', msg)
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

    def is_ok(self):
        return self.status() == 'SKIP' or self.status() == 'PASS'


class Result:

    def __init__(self, root, path):
        self.path = path
        # The current results
        self.cur = RawResult(root / path)
        # The last results matching the platform of the current results
        self.ref = RawResult(GOLDEN_ROOT / path)

        self.ratio = None
        if self.cur.execution_duration and self.ref.execution_duration:
            self.ratio = self.ref.execution_duration / self.cur.execution_duration

    def __repr__(self):
        return '<Result({})>'.format(self.path)

    def check_result(self, skip, compare, precision, perf_threshold, expected, correct):
        print(self.path, self.cur.compile_duration, self.ref.execution_duration,
              self.cur.execution_duration, self.ratio)

        if not self.cur.exists():
            print('  missing cur')
        if not compare and not self.ref.exists():
            print('  missing ref')

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
                    test_result.add_error('Missing golden duration')
                elif not self.cur.execution_duration:
                    test_result.add_error('Missing result duration')
                else:
                    if self.ratio < perf_threshold:
                        test_result.add_failure('Performance regression')

                base_output = self.ref.np_data
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('path', type=Path)
    parser.add_argument('--skip', type=bool, default=False)
    parser.add_argument('--compare', type=bool, default=True)
    parser.add_argument('--precision', choices=['untested', 'high', 'low'], default='untested')
    parser.add_argument('--threshold', type=float, default=DEFAULT_PERF_THRESHOLD)
    parser.add_argument('--expected', type=str)
    parser.add_argument('--correct', type=bool, default=True)
    args = parser.parse_args()

    build_url = os.getenv('BUILDKITE_BUILD_URL')
    if build_url:
        job_id = os.getenv('BUILDKITE_JOB_ID')
        build_url = f'{build_url}#{job_id}'
    else:
        build_url = DEFAULT_BUILD_URL

    result = Result(args.root, args.path)
    test_result = result.check_result(args.skip, args.compare, args.precision, args.threshold,
                                      args.expected, args.correct)
    report = {
        'build_url': build_url,
        'compare': test_result.compare,
        'errors': test_result.errors,
        'failures': test_result.failures,
        'ratio': result.ratio,
        'reason': test_result.reason(),
        'status': test_result.status(),
        'compile_duration': result.cur.compile_duration,
        'cur.execution_duration': result.cur.execution_duration,
        'ref.execution_duration': result.ref.execution_duration,
    }

    with (args.root / args.path / 'report.json').open('w') as fp:
        json.dump(report, fp)

    if not test_result.is_ok():
        sys.exit(1)


if __name__ == '__main__':
    main()
