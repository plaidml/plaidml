# Copyright 2017-2018 Intel Corporation.

import argparse
import os
import sys
from collections import OrderedDict

import pystache
import yaml

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

if PY3:
    string_types = str
else:
    string_types = basestring


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """Helper function to make YAML maps retain order in python"""

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                                  construct_mapping)
    return yaml.load(stream, OrderedLoader)


class Node(object):
    """Class to represent node object when read from YAML"""

    def __init__(self, parent, name, yaml_map):
        self.parent = parent
        self.name = name
        self.elements = yaml_map
        for (field, user_type) in self.elements.iteritems():
            if not isinstance(field, string_types):
                raise Exception("Field name must be a string in node type " + name + ", not: " +
                                str(field))
            if not isinstance(user_type, string_types):
                raise Exception("Field type must be a string in node type " + name + ", not: " +
                                str(user_type))
            self.elements[field] = user_type

    def get_node_names(self):
        return set([self.name])

    def get_group_names(self):
        return set()

    def get_nodes(self, node_names, group_names):
        elements = []
        for (field, otype) in self.elements.iteritems():
            element = {}
            is_primitive = not (otype in group_names or otype in node_names)
            element['field'] = field
            element['user_type'] = otype
            element['is_primitive'] = is_primitive
            if otype in group_names:
                element['hold_type'] = otype + "Object"
            elif otype in node_names:
                element['hold_type'] = otype + "Object"
            else:
                element['hold_type'] = "PrimitiveWrapper<" + otype + ">"
            elements.append(element)
            element['comma'] = ','
        elements[-1]['comma'] = ''
        return [{'node': self.name, 'group': self.parent, 'elements': elements}]

    def get_groups(self):
        return []

    def get_primitives(self, node_names, group_names):
        primitives = set()
        for (field, otype) in self.elements.iteritems():
            if not (otype in group_names or otype in node_names):
                primitives.add(otype)
        return primitives


class Group(object):
    """Class to represent group (subtype) objects when read from yaml file"""

    def __init__(self, parent, name, yaml_map):
        self.parent = parent
        self.name = name
        self.subtypes = OrderedDict()
        for (subtype, submap) in yaml_map.iteritems():
            if not isinstance(subtype, string_types):
                raise Exception("Subtype name must be a string in group type " + subtype +
                                ", not: " + str(subtype))
            if not isinstance(submap, dict):
                raise Exception("All types must be a yaml map, invalid type " + subtype +
                                ", not: " + str(submap))
            if len(submap) == 0:
                raise Exception("Empty types not allowed: " + subtype)
            if isinstance(submap.itervalues().next(), string_types):
                self.subtypes[subtype] = Node(name, subtype, submap)
            else:
                self.subtypes[subtype] = Group(name, subtype, submap)

    def get_node_names(self):
        return set.union(*[subtype.get_node_names() for subtype in self.subtypes.values()])

    def get_group_names(self):
        return set.union(
            set([self.name]), *[subtype.get_group_names() for subtype in self.subtypes.values()])

    def get_nodes(self, node_names, group_names):
        return sum(
            [subtype.get_nodes(node_names, group_names) for subtype in self.subtypes.values()], [])

    def get_groups(self):
        out_group = [{'group': self.name, 'parent': self.parent}]
        return out_group + sum([subtype.get_groups() for subtype in self.subtypes.values()], [])

    def get_primitives(self, node_names, group_names):
        return set.union(*[
            subtype.get_primitives(node_names, group_names) for subtype in self.subtypes.values()
        ])


def get_list_of_strings(defs, n):
    """A function that verifies a list of strings"""
    x = defs.get(n)
    if not isinstance(x, list):
        raise Exception("Expecting " + n + " to be a list of strings, got: " + str(x))
    for s in x:
        if not isinstance(s, string_types):
            raise Exception("Expecting " + n + " to be a list of strings, got: " + str(x))
    return x


def translate(defs):
    """Transform the data from the raw YAML form to something easy to use in templates"""
    if not isinstance(defs, dict):
        raise Exception("Overall config must be a dictionary")

    # Get the c++ goo we need
    namespace = get_list_of_strings(defs, 'Namespace')
    system_includes = get_list_of_strings(defs, 'SystemIncludes')
    user_includes = get_list_of_strings(defs, 'UserIncludes')

    # Get the name of the toplevel node (if not node)
    toplevel = defs.get('Toplevel', None)
    if not isinstance(toplevel, dict):
        raise Exception("Toplevel must be a map")
    if len(toplevel) != 1:
        raise Exception("Toplevel must have a single entry")
    top_name = toplevel.keys()[0]
    top_map = toplevel[top_name]
    if not isinstance(top_map, dict):
        raise Exception("Toplevel typename does not refer to a yaml map")

    # Extract all the actual typing information
    types = Group(None, top_name, top_map)
    node_names = types.get_node_names()
    group_names = types.get_group_names()
    groups = types.get_groups()
    nodes = types.get_nodes(node_names, group_names)
    primitives = types.get_primitives(node_names, group_names)

    system_builtins = ['memory', 'type_traits', 'unordered_map', 'vector', 'string']

    # Reformat it for the templating goo
    everything = {
        'namespace': namespace,
        'revnamespace': reversed(namespace),
        'systeminc': sorted(list(set(system_includes + system_builtins))),
        'userinc': sorted(list(set(user_includes + ['base/util/intern.h']))),
        'toplevel': top_name,
        'nodes': nodes,
        'groups': groups,
        'primitives': list(primitives),
    }

    # Return the results
    return everything


def main():
    # Parse the command line
    parser = argparse.ArgumentParser(description='Build AST classes')
    parser.add_argument('-i', help='YAML input file describing AST')
    parser.add_argument('-t', default="base", help='Template file to use')
    parser.add_argument('-o', help='Location to write c++ output header')
    args = parser.parse_args()

    # Load the template
    code_path = os.path.dirname(__file__)
    with open(os.path.join(code_path, args.t + ".template"), 'r') as f:
        template = f.read()

    # Parse the input YAML, retaining order as an OrderedDict
    with open(args.i, 'r') as f:
        defs = ordered_load(f, yaml.SafeLoader)

    # Render the output
    output = ""
    output += pystache.render(template, translate(defs))

    with open(args.o, 'w') as f:
        f.write(output)


if __name__ == "__main__":
    main()
