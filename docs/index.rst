

.. image:: https://travis-ci.org/plaidml/plaidml.svg?branch=master
   :alt: Build Status

PlaidML is an advanced and portable tensor compiler for enabling deep learning 
on laptops, embedded devices, or other kinds of workstations where compute may be 
limited.

As a component within the `nGraph Compiler stack`_, PlaidML further extends the 
capabilities of specialized deep-learning hardware (especially GPUs,) and makes 
it both easier and faster to access or make use of subgraph-level optimizations 
that would otherwise be bounded by the compute limitations of the device. The 
latest version of PlaidML includes initial support for **PlaidML**'s ``Stripe``, 
written in the :doc:`Tile <tile/about>` language. Tile was specifically created 
to be able to overcome the many limitations of fussy, brittle and tightly-coupled 
GPU-based software architectures. By bridging the gap between the universal 
mathematical descriptions of deep learning operations, such as convolution, and 
the platform and chip-specific operations that enable acceleration, PlaidML makes 
deep learning work 

As a component under `Keras`_ PlaidML can accelerate training workloads with 
customized or automatically-generated Tile code. It works especially well on 
GPUs, and it doesn't require use of CUDA/cuDNN on Nvidia* hardware, while 
achieving comparable performance.

It works on all major operating systems: Linux, `macOS`_, and `Windows`_. 

For background and early benchmarks, see our `blog post`_.

PlaidML is under active development and should be thought of as alpha quality.

.. toctree::
   :maxdepth: 1
   :caption: PlaidML 

   Installing <install>
   Building <building>
   API Reference <plaidml>
   
   
.. toctree::
   :maxdepth: 2
   :caption: Understanding Tile
   
   tile/index   

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   building_a_frontend
   writing_tile_code
   adding_ops


.. toctree::
   :maxdepth: 1
   :caption: Project Metadata
   
   Architecture Overview <overview>
   release-notes
   Contributing <contributing>

   


License
-------

PlaidML is licensed under `Apache2`_.


Reporting Issues
----------------

`Open a ticket on GitHub`_ or `post to plaidml-dev`_. 


Indices and tables
==================

   * :ref:`search`   
   * :ref:`genindex`




.. _macOS: http://vertex.ai/blog/plaidml-mac-preview
.. _Windows: http://vertex.ai/blog/deep-learning-for-everyone-plaidml-for-windows
.. _blog post: https://ai.intel.com/reintroducing-plaidml/
.. _nGraph compiler stack: http://ngraph.nervanasys.com/docs/latest/
.. _Open a ticket on GitHub: https://github.com/plaidml/plaidml/issues
.. _post to plaidml-dev: https://groups.google.com/forum/#!forum/plaidml-dev
.. _Apache2: https://raw.githubusercontent.com/plaidml/plaidml/master/LICENSE
.. _Keras: https://keras.io/