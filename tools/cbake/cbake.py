# Copyright 2019 Intel Corporation.

import argparse
from pathlib import Path

import cmake_parser


class Block:

    def __init__(self, parent=None, cmd=None):
        self.cmd = cmd
        self.parent = parent
        self.kids = []
        self.body = []

    def push(self, cmd):
        kid = Block(self, cmd)
        self.kids.append(kid)
        return kid

    def pop(self):
        return self.parent


class Command:

    def __init__(self, cmd):
        self.name = cmd.name
        self.args = [x.contents for x in cmd.body]

    def __repr__(self):
        return 'Command<{}({})>'.format(self.name, self.args)

    def run(self, vm):
        pass


class Set(Command):

    def __init__(self, cmd):
        self.name = cmd.body[0].contents
        self.values = [x.contents for x in cmd.body[1:]]

    def __repr__(self):
        return 'Set<{} = {}>'.format(self.name, self.values)


class Function(Command):

    def __init__(self, cmd, body):
        self.name = cmd.body[0].contents
        self.args = [x.contents for x in cmd.body[1:]]
        self.body = body

    def __repr__(self):
        return 'Function<{}({})>'.format(self.name, self.args)


class ConditionalArm(Command):

    def __init__(self, cmd):
        self.cond = [x.contents for x in cmd.body]
        self.body = None

    def __repr__(self):
        return str(self.cond)


class Conditional(Command):

    def __init__(self, cmd):
        self.if_cond = ConditionalArm(cmd)
        self.elseif_conds = []
        self.else_cond = None
        self.cur = self.if_cond

    def __repr__(self):
        return 'If<{}, {}, {}>'.format(self.if_cond, self.elseif_conds, self.else_cond)

    def add_elseif(self, cmd, block):
        self.cur.body = block
        self.cur = ConditionalArm(cmd)
        self.elseif_conds.append(self.cur)

    def add_else(self, cmd, block):
        self.cur.body = block
        self.else_cond = self.cur = ConditionalArm(cmd)

    def endif(self, cmd, block):
        self.cur.body = block


class ForEach(Command):

    def __init__(self, cmd, body):
        self.var = cmd.body[0].contents
        self.items = [x.contents for x in cmd.body[1:]]
        self.body = body

    def __repr__(self):
        return 'ForEach<{} in {}>'.format(self.var, self.items)


class AddSubdirectory(Command):

    def __init__(self, cmd):
        self.dir = cmd.body[0].contents

    def __repr__(self):
        return 'AddSubdirectory<{}>'.format(self.dir)

    def run(self, vm):
        vm.add_subdirectory(self.dir)


class Structure:

    def __init__(self):
        self.cmds = {
            'add_subdirectory': self.cmd_add_subdirectory,
            'set': self.cmd_set,
            'function': self.cmd_function,
            'endfunction': self.cmd_endfunction,
            'if': self.cmd_if,
            'elseif': self.cmd_elseif,
            'else': self.cmd_else,
            'endif': self.cmd_endif,
            'foreach': self.cmd_foreach,
            'endforeach': self.cmd_endforeach,
            'macro': self.cmd_macro,
            'endmacro': self.cmd_endmacro,
        }
        self.cur_block = self.top = Block()

    def command(self, cmd):
        fn = self.cmds.get(cmd.name.lower())
        if fn:
            fn(cmd)
        else:
            self.add_cmd(Command(cmd))

    def add_cmd(self, cmd):
        self.cur_block.body.append(cmd)

    def cmd_set(self, cmd):
        self.add_cmd(Set(cmd))

    def cmd_function(self, cmd):
        self.cur_block = self.cur_block.push(cmd)

    def cmd_endfunction(self, cmd):
        fn_cmd = self.cur_block.cmd
        assert fn_cmd.name == 'function'
        body = self.cur_block
        self.cur_block = self.cur_block.pop()
        self.add_cmd(Function(fn_cmd, body))

    def cmd_if(self, cmd):
        self.cur_block = self.cur_block.push(Conditional(cmd))

    def cmd_elseif(self, cmd):
        cond = self.cur_block.cmd
        assert isinstance(cond, Conditional)
        body = self.cur_block
        cond.add_elseif(cmd, body)
        self.cur_block = self.cur_block.pop()
        self.cur_block = self.cur_block.push(cond)

    def cmd_else(self, cmd):
        cond = self.cur_block.cmd
        assert isinstance(cond, Conditional)
        body = self.cur_block
        cond.add_else(cmd, body)
        self.cur_block = self.cur_block.pop()
        self.cur_block = self.cur_block.push(cond)

    def cmd_endif(self, cmd):
        cond = self.cur_block.cmd
        assert isinstance(cond, Conditional)
        body = self.cur_block
        cond.endif(cmd, body)
        self.cur_block = self.cur_block.pop()
        self.add_cmd(cond)

    def cmd_foreach(self, cmd):
        self.cur_block = self.cur_block.push(cmd)

    def cmd_endforeach(self, cmd):
        foreach_cmd = self.cur_block.cmd
        assert foreach_cmd.name.lower() == 'foreach'
        body = self.cur_block
        self.cur_block = self.cur_block.pop()
        self.add_cmd(ForEach(foreach_cmd, body))

    def cmd_macro(self, cmd):
        self.cur_block = self.cur_block.push(cmd)

    def cmd_endmacro(self, cmd):
        prev_cmd = self.cur_block.cmd
        assert prev_cmd.name == 'macro'
        body = self.cur_block
        self.cur_block = self.cur_block.pop()
        self.add_cmd(Command(prev_cmd))

    def cmd_add_subdirectory(self, cmd):
        self.add_cmd(AddSubdirectory(cmd))

    def cmd_skip(self, cmd):
        # print('skipping {}'.format(cmd.name))
        pass


class VirtualMachine:

    def __init__(self, path, vars={}):
        self.vars = vars
        self.cwd = path.resolve()
        self.cmakelists_path = self.cwd / 'CMakeLists.txt'
        print('Processing {}'.format(self.cmakelists_path))

    def add_subdirectory(self, subdir):
        vm = VirtualMachine(self.cwd / subdir)
        vm.run()

    def run(self):
        cmds = parse_file(self.cmakelists_path.read_text())
        for cmd in cmds:
            print(cmd)
            cmd.run(self)


def parse_file(path):
    ptree = cmake_parser.parse(path)
    struct = Structure()
    for cmd in ptree:
        if not isinstance(cmd, cmake_parser._Command):
            continue
        struct.command(cmd)
    return struct.top.body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    args = parser.parse_args()

    vm = VirtualMachine(args.input)
    vm.run()


if __name__ == "__main__":
    main()
