# Copyright Vertex.AI

import os


def config():
  filename = os.getenv('PLAIDML_CONFIG', '../vertexai_plaidml/testing/tile_generic_cpu.json')
  with open(filename) as file_:
    return file_.read()


def very_large_values_config():
  return """
{
  "platform": {
    "@type": "type.vertex.ai/vertexai.tile.local_machine.proto.Platform",
    "hals": [
      {
        "@type": "type.vertex.ai/vertexai.tile.hal.opencl.proto.Driver",
        "sel": {
          "value": "true"
        }
      }
    ],
    "hardware_configs": [
      {
        "sel": {
          "value": "true"
        },
        "settings": {
          "max_mem": {"value": 1000000},
          "max_regs": {"value": 1000000},
          "mem_width": {"value": 1000000},
          "threads": {"value": 1000000},
          "vec_size": {"value": 1000000},
          "enable_half": {"value": "false"}
        }
      }
    ]
  }
}
"""
