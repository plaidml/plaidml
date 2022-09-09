import itertools
import unittest
from pathlib import Path

import yaml


class Parameter:

    def __init__(self, key, value):
        self.key = key
        if isinstance(value, list):
            self.value = value
        else:
            self.value = [value]

    def __repr__(self):
        return f'({self.key}, {self.value})'

    def tuples(self):
        return [(self.key, x) for x in self.value]


def has_params(obj, selector):
    params = {k: Parameter(k, v) for k, v in obj['params'].items()}
    for k, v in selector.items():
        p = params.get(k)
        if not p:
            return False
        if v not in p.value:
            return False
    return True


class Action:

    def __init__(self, plan, src, case, ctx, root=Path('.')):
        self.src = src
        self.case = case
        self.vars = plan.vars.copy()
        plan.import_vars(case.params, self.vars)
        self.vars.update(case.params)
        self.vars.update(case.rule.vars)
        self.vars.update(path=case.path(root))
        self.name = self.case.rule.name.format(**self.vars)
        self.vars['name'] = self.name
        self.expand(self.case.rule.expand)
        self.expand(ctx)
        self.cmds = [x.format(**self.vars) for x in src]

    def expand(self, ctx):
        for k, v in ctx.items():
            if isinstance(v, list):
                self.vars[k] = [x.format(**self.vars) for x in v]
            else:
                self.vars[k] = str(v).format(**self.vars)


class Case:

    def __init__(self, rule, params):
        self.rule = rule
        self.params = params
        self.name = self.rule.name.format(**params)

    def __repr__(self):
        return self.name

    def path(self, root=Path('.')):
        return Path(root, *self.name.split('/'))


class Rule:

    def __init__(self, src, selector={}):
        self.src = src
        self.name = src['name']
        self.vars = src.get('vars', {})
        self.expand = src.get('expand', {})
        self.options = src.get('options')
        self.actions = src.get('actions', [])
        self.params = src['params'].copy()
        for k, v in selector.items():
            self.params[k] = v

    def __repr__(self):
        return self.name

    def product(self):
        params = [Parameter(k, v) for k, v in self.params.items()]
        product = itertools.product(*[x.tuples() for x in params])
        return [Case(self, dict(x)) for x in product]


class Plan:

    def __init__(self, doc):
        self.doc = doc
        self.vars = doc.get('VARS', {})
        self.params = doc.get('PARAMS', {})
        self.actions = doc.get('ACTIONS', [])
        self.rules = doc.get('RULES', [])
        self.settings = doc.get('SETTINGS', [])

    def get_rules(self, selector):
        return [Rule(x, selector) for x in self.rules if has_params(x, selector)]

    def get_settings(self, selector):
        return [x for x in self.settings if has_params(x, selector)]

    def get_cases(self, selector):
        for rule in self.get_rules(selector):
            for case in rule.product():
                yield case

    def get_actions(self, selector, root=Path('.')):
        settings = {}
        for setting in self.get_settings(selector):
            settings.update(setting['select'])

        for case in self.get_cases(selector):
            for ref in case.rule.actions:
                ctx = settings.get(case.name, {})
                yield Action(self, self.actions.get(ref), case, ctx, root=root)

    def import_vars(self, params, into):
        for pkey, pval in params.items():
            pobj = self.params.get(pkey, {}).get(pval, {})
            into.update(pobj.get('vars', {}))
            self.import_vars(pobj.get('imports', {}), into)


def load(path):
    with open(path) as fp:
        return Plan(yaml.safe_load(fp))


class TestPlan(unittest.TestCase):

    PLAN = '''
VARS:
  version: 1.0.0

PARAMS:
  build:
    linux_x86_64:
      vars:
        arch: manylinux1_x86_64
        build_emoji: ":linux:"
        build_root: build-x86_64
  cfg:
    release:
      vars:
        build_type: Release
  platform:
    pml-llvm-cpu:
      imports:
        build: linux_x86_64
        cfg: release
      vars:
        platform_emoji: ":plaidml::crown:"
        depends_on: linux_x86_64/release

ACTIONS:
  plaidbench:
    - buildkite-agent artifact download {build_root}/{build_type}/plaidml-{version}-py3-none-{arch}.whl .
    - python ci/runners/plaidbench.py --examples={examples} --batch-size={batch_size} --results={path} keras {model}

RULES:
  - name: "{platform}/{suite}/{model}/bs={batch_size}"
    params:
      pipeline: [nightly, plaidml]
      suite: infer
      platform: pml-llvm-cpu
      batch_size: [1]
      model:
        - inception_v3
        - mobilenet
    actions: [plaidbench]
    vars:
      examples: 1024
      timeout: 10
      artifacts:
        - tmp/test/**/report.json
    expand:
      emoji: "{build_emoji}{platform_emoji}"

SETTINGS:
  - params:
      pipeline: [nightly, plaidml]
    select:
      pml-llvm-cpu/infer/inception_v3/bs=1: {examples: 64}
'''

    def test_cases(self):
        plan = Plan(yaml.safe_load(self.PLAN))
        cases = list(plan.get_cases(dict(pipeline='plaidml')))
        self.assertSequenceEqual([str(x) for x in cases], [
            'pml-llvm-cpu/infer/inception_v3/bs=1',
            'pml-llvm-cpu/infer/mobilenet/bs=1',
        ])
        self.assertSequenceEqual([x.path() for x in cases], [
            Path('pml-llvm-cpu') / 'infer' / 'inception_v3' / 'bs=1',
            Path('pml-llvm-cpu') / 'infer' / 'mobilenet' / 'bs=1',
        ])

    def test_actions(self):
        plan = Plan(yaml.safe_load(self.PLAN))
        selector = dict(pipeline='plaidml')
        actions = list(plan.get_actions(selector, root=Path('tmp')))
        self.assertSequenceEqual([x.cmds for x in actions], [
            [
                'buildkite-agent artifact download build-x86_64/Release/plaidml-1.0.0-py3-none-manylinux1_x86_64.whl .',
                'python ci/runners/plaidbench.py --examples=64 --batch-size=1 --results=tmp/pml-llvm-cpu/infer/inception_v3/bs=1 keras inception_v3',
            ],
            [
                'buildkite-agent artifact download build-x86_64/Release/plaidml-1.0.0-py3-none-manylinux1_x86_64.whl .',
                'python ci/runners/plaidbench.py --examples=1024 --batch-size=1 --results=tmp/pml-llvm-cpu/infer/mobilenet/bs=1 keras mobilenet',
            ],
        ])
        self.assertSequenceEqual([x.vars['emoji'] for x in actions], [
            ':linux::plaidml::crown:',
            ':linux::plaidml::crown:',
        ])
        self.assertSequenceEqual([x.vars['artifacts'] for x in actions], [
            ['tmp/test/**/report.json'],
            ['tmp/test/**/report.json'],
        ])


if __name__ == '__main__':
    unittest.main()
