These tests demonstrate the lowering of TensorFlow (XLA HLO) via PlaidML.
# Types of Tests
## Unit Tests
Unit tests can be written in one of two ways:
- Use the `xla.builder` API
- Explicit HLO Specification
## Full-Sized Network Tests
Full-sized networks contain pretrained weights that should be passed directly to
the test. There is a multi-step process that will preserve weights for tests and
reference implementations: 
1. Load a TensorFlow 1.x Saved Model 
    - TensorFlow Hub Networks  
    Networks available through [TensorFlow Hub](tfhub.dev) can be instantiated
    using the [`hub.Module`](https://www.tensorflow.org/hub/api_docs/python/hub/Module)
    API.
    - Non-TensorFlow Hub Networks  
    If you have a pretrained model available locally, it can be loaded using the
    [`tf.saved_model.load`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/load)
    API.  
2. Freeze the network
   Networks can be frozen using the
   [`tf.graph_util.convert_weights_to_constants`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/graph_util/convert_variables_to_constants)
   API. Freezing the network allows weights to be embedded into the model
   directly as constants. Note that this feature is currently only available in
   TF 1.x.
3. Convert the network into XLA HLO Protobuf
  There are two environment variables, `XLA_FLAGS` and `TF_XLA_FLAGS`, which
  should be specified in order to convert a network to the XLA HLO Protobuf
  format. `TF_XLA_FLAGS` tells TensorFlow to use XLA, and `XLA_FLAGS` tells XLA
  to write a HLO Protobuf for use in the tests:  
  ```XLA_FLAGS "--xla_dump_to=/your/dump/location --xla_dump_hlo_as_proto"
  TF_XLA_FLAGS = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"```
# Validated Networks
Below is a list of TensorFlow networks validated via the PlaidML backend:
 - I3D
 - ResNext-50
 - ResNet-152
## Methodology
Each validated network has a reference implementation, which is a frozen graph
that is used for correctness and performance baselines.  
### Assessing Correctness
Correctness is assessed by comparing the output results of an inference
performed via TensorFlow and the same inference performed via PlaidML.  
### Measuring Performance
Performance is assessed by comparing the inference time via TensorFlow and the
inference time via PlaidML.  
#### TensorFlow Inference Time
In TensorFlow 1.x, inferences are run via the
[`Session`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session)
API. The execution time can be measured as the time it takes to [run a
`Session`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session#run).
#### PlaidML Inference Time
In PlaidML, inferences are run via the
[`Executable`](https://github.com/plaidml/plaidml/blob/plaidml-v1/plaidml/exec/exec.h)
API. The execution time can be measured as the time it takes to [run an
`Executable`](https://github.com/plaidml/plaidml/blob/plaidml-v1/plaidml/exec/exec.h#L83-L100).
