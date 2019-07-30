import json
import pathlib

import numpy as np
import util

DEFAULT_RATIO_THRESHOLD = 0.8
GOLDEN_ROOT = pathlib.Path('ci/golden')


class RawResult(object):

    def __init__(self, root, test_info):
        self.path = test_info.path(root)

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
                util.printf('  Exception:', ex)
                if self.exception is None:
                    self.exception = str(ex)

    def exists(self):
        return self.path.exists()


class TestResult(object):

    def __init__(self, skip, compare):
        self.errors = []
        self.failures = []
        self.skips = []
        self.expected = None
        self.skipped = skip
        self.compare = compare

    def add_error(self, msg):
        util.printf('  ERROR:', msg)
        self.errors.append(msg)

    def add_failure(self, msg):
        util.printf('  FAIL:', msg)
        self.failures.append(msg)

    def add_skip(self, msg):
        util.printf('  SKIP:', msg)
        self.skips.append(msg)

    def set_expected(self, msg):
        util.printf('  SKIP:', msg)
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


class Result(object):

    def __init__(self, root, test_info, golden_info):
        self.test_info = test_info
        # The current results
        self.cur = RawResult(root, test_info)
        # The last results matching the platform of the current results
        self.ref = RawResult(GOLDEN_ROOT, test_info)
        # The golden results for the baseline platform (usually a TF/CUDA variant)
        self.golden = RawResult(GOLDEN_ROOT, golden_info)

        self.ratio = None
        if self.cur.execution_duration and self.ref.execution_duration:
            self.ratio = self.ref.execution_duration / self.cur.execution_duration

        cur_efficiency = None
        base_efficiency = None
        if self.cur.execution_duration and self.test_info.platform.gpu_flops:
            cur_efficiency = self.cur.execution_duration * self.test_info.platform.gpu_flops
        if self.golden.execution_duration:
            base_efficiency = self.golden.execution_duration * golden_info.platform.gpu_flops

        self.efficiency = None
        if cur_efficiency and base_efficiency:
            self.efficiency = base_efficiency / cur_efficiency

        label_parts = [self.test_info.platform.gpu, self.test_info.workload_name]
        if self.test_info.batch_size:
            label_parts += [str(self.test_info.batch_size)]
        self.label = '-'.join(label_parts)

        self.test_result = self._check_result()

    def __repr__(self):
        return '<Result({})>'.format(self.test_info)

    def _check_result(self):
        util.printf(self.test_info, self.cur.compile_duration, self.ref.execution_duration,
                    self.cur.execution_duration, self.ratio, self.efficiency)

        skip = self.test_info.workload.get('skip', False)
        expected = self.test_info.workload.get('expected')
        precision = self.test_info.workload.get('precision')
        perf_threshold = self.test_info.workload.get('perf_threshold', DEFAULT_RATIO_THRESHOLD)
        correct = self.test_info.workload.get('correct', True)
        popt = util.PlanOption(self.test_info.suite, self.test_info.workload,
                               self.test_info.platform)
        compare = popt.get('compare', True)

        if not self.cur.exists():
            util.printf('  missing cur')
        if not compare and not self.ref.exists():
            util.printf('  missing ref')

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
