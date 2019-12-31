# Copyright 2019 Intel Corporation
# Copyright 2015 Open Source Robotics Foundation, Inc.
# Copyright 2013 Willow Garage, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based on: https://github.com/ijt/cmakelists_parsing

from collections import namedtuple
import re

_Arg = namedtuple('Arg', 'contents comments')
_Command = namedtuple('Command', 'name body comment')
BlankLine = namedtuple('BlankLine', '')


class File(list):
    """Top node of the syntax tree for a CMakeLists file."""

    def __repr__(self):
        return 'File(' + repr(list(self)) + ')'


class Comment(str):

    def __repr__(self):
        return 'Comment(' + str(self) + ')'


def Arg(contents, comments=None):
    return _Arg(contents, comments or [])


def Command(name, body, comment=None):
    return _Command(name, body, comment)


class CMakeParseError(Exception):
    pass


def parse(s, path='<string>'):
    '''
    Parses a string s in CMakeLists format whose
    contents are assumed to have come from the
    file at the given path.
    '''
    nums_toks = tokenize(s)
    nums_items = list(parse_file(nums_toks))
    items = [item for _, item in nums_items]
    return File(items)


def parse_file(toks):
    '''
    Yields line number ranges and top-level elements of the syntax tree for
    a CMakeLists file, given a generator of tokens from the file.

    toks must really be a generator, not a list, for this to work.
    '''
    prev_type = 'NEWLINE'
    for line_num, (typ, tok_contents) in toks:
        if typ == 'COMMENT':
            yield ([line_num], Comment(tok_contents))
        elif typ == 'NEWLINE' and prev_type == 'NEWLINE':
            yield ([line_num], BlankLine())
        elif typ == 'WORD':
            line_nums, cmd = parse_command(line_num, tok_contents, toks)
            yield (line_nums, cmd)
        prev_type = typ


def parse_command(start_line_num, command_name, toks):
    cmd = Command(name=command_name, body=[], comment=None)
    expect('LPAREN', toks)
    paren_sum = 1
    for line_num, (typ, tok_contents) in toks:
        if typ == 'RPAREN':
            paren_sum -= 1
            if paren_sum == 0:
                line_nums = range(start_line_num, line_num + 1)
                return line_nums, cmd
        elif typ == 'LPAREN':
            paren_sum += 1
        elif typ in ('WORD', 'STRING'):
            cmd.body.append(Arg(tok_contents, []))
        elif typ == 'COMMENT':
            c = tok_contents
            if cmd.body:
                cmd.body[-1].comments.append(c)
            else:
                cmd.comments.append(c)
    msg = 'File ended while processing command "%s" started at line %s' % (command_name,
                                                                           start_line_num)
    raise CMakeParseError(msg)


def expect(expected_type, toks):
    line_num, (typ, tok_contents) = next(toks)
    if typ != expected_type:
        msg = 'Expected a %s, but got "%s" at line %s' % (expected_type, tok_contents, line_num)
        raise CMakeParseError(msg)


# http://stackoverflow.com/questions/691148/pythonic-way-to-implement-a-tokenizer
# TODO: Handle multiline strings.
scanner = re.Scanner([
    (r'#.*', lambda scanner, token: ("COMMENT", token)),
    (r'"[^"]*"', lambda scanner, token: ("STRING", token)),
    (r"\(", lambda scanner, token: ("LPAREN", token)),
    (r"\)", lambda scanner, token: ("RPAREN", token)),
    (r'[^ \t\r\n()#"]+', lambda scanner, token: ("WORD", token)),
    (r'\n', lambda scanner, token: ("NEWLINE", token)),
    (r"\s+", None),  # skip other whitespace
])


def tokenize(s):
    """
    Yields pairs of the form (line_num, (token_type, token_contents))
    given a string containing the contents of a CMakeLists file.
    """
    toks, remainder = scanner.scan(s)
    line_num = 1
    if remainder != '':
        msg = 'Unrecognized tokens at line %s: %s' % (line_num, remainder)
        raise ValueError(msg)
    for tok_type, tok_contents in toks:
        yield line_num, (tok_type, tok_contents.strip())
        line_num += tok_contents.count('\n')
