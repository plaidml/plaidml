.. image:: images/plaid-final.png
   :alt: The PlaidML Platypus

*A framework for making deep learning work everywhere.*

.. image:: https://travis-ci.org/plaidml/plaidml.svg?branch=master
   :alt: Build Status

PlaidML is a multi-language acceleration framework that:

* Enables practitioners to deploy high-performance neural nets on any device
* Allows hardware developers to quickly integrate with high-level frameworks
* Allows framework developers to easily add support for many kinds of hardware
* Works on all major platforms - Linux, `macOS <http://vertex.ai/blog/plaidml-mac-preview>`_,
  `Windows <http://vertex.ai/blog/deep-learning-for-everyone-plaidml-for-windows>`_

For background and early benchmarks see our
`blog post <http://vertex.ai/blog/announcing-plaidml>`_ announcing the
release. PlaidML is under active development and should be thought of as
alpha quality.

.. toctree::
   :maxdepth: 1

   Installing <installing>
   Building <building>
   Architecture Overview <overview>
   API Reference <plaidml>
   life-of-a-tile-function
   Contributing <contributing>

License
-------

PlaidML is licensed under the
`AGPLv3 <https://www.gnu.org/licenses/agpl-3.0.txt>`_.

Our open source goals include 1) helping students get started with deep
learning as easily as possible and 2) helping researchers develop new methods
more quickly than is possible with other tools. PlaidML is unique in being
fully open source and free of dependence on libraries like cuDNN that carry
revocable and redistribution-prohibiting licenses. For situations where an
alternate license is preferable please contact
`solutions@vertex.ai <mailto:solutions@vertex.ai>`_.

Reporting Issues
----------------

Either open a ticket on `GitHub <https://github.com/plaidml/plaidml/issues>`_ or post to `plaidml-dev <https://groups.google.com/forum/#!forum/plaidml-dev>`_.

