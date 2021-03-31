Getting Started
###############

Select the tab which pertains to your use case:


.. tabs::

   .. group-tab:: Quick Install (via pip)
      Installation Instructions
      #########################
      PlaidML supports :ref:`Ubuntu`, :ref:`macOS` , and :ref:`Microsoft Windows` operating systems.
      
      .. tabs::
      
          .. tab:: Ubuntu
      
              If necessary, install Python's ``pip`` tool. OpenCL 1.2 or greater is alsorequired.
      
              .. code-block::
      
                  sudo add-apt-repository universe && sudo apt update
                  sudo apt install python3-pip
                  sudo apt install clinfo
      
              Run ``clinfo``, and if it reports ``"Number of platforms" == 0``, you can install a driver (GPU) or enable a CPU via one of these options:
      
              *  **Nvidia** -- For Nvidia GPUs, run:
      
              .. code-block::
      
                  sudo add-apt-repository ppa:graphics-drivers/ppa && sudo apt update
                  sudo apt install nvidia-modprobe nvidia-384 nvidia-opencl-icd-384 libcuda1-384
      
              *  **AMD** -- For AMD graphics cards, `download the AMDGPU PRO driver <http://support.amd.com/en-us/kb-articles/Pages/AMDGPU-PRO-Driver-for-Linux-Release-Notes.aspx>`_ and follow the 
                 instructions provided by AMD for the chip.
      
              *  **Intel® Xeon® Processors OR Intel® Core™ Processors** -- In lieu of installing specific drivers, 
                 you can `install ngraph with pip <https://github.com/NervanaSystems/ngraph/blob/master/README.md#quick-start>`_, or you can `build the nGraph Library <https://ngraph.nervanasys.com/docs/latest/buildlb.html>`_ with the 
                 cmake flag `-DNGRAPH*PLAIDML*ENABLE=TRUE`.
      
              **Python**
              
              Although PlaidML can be run with Python2, we recommend Python3, as well as judicious use of a `Virtualenv <https://virtualenv.pypa.io/en/stable>`_.  To create one just for using PlaidML:
      
              .. code-block::
      
                  python3 -m venv plaidml-venv
                  source plaidml-venv/bin/activate
      
              **Keras**
             
              There are two ways to get Keras working on your system:
      
              1. Isolate it to your `venv` as follows:
      
              .. code-block:: shell
      
                  pip install -U plaidml-keras
      
              2. Alternatively, install the PlaidML wheels system-wide with:
      
              .. code-block::
      
                  sudo -H pip install -U plaidml-keras
      
              Finally, set up PlaidML to use a preferred computing device:
      
              .. code-block::
      
                  plaidml-setup
      
              You can test the installation by running MobileNet in `plaidbench <https://github.com/plaidml/plaidml/tree/plaidml-v1/plaidbench>`_. Remember to use ``sudo -H`` if you're working outside of a virtual environment.
      
              .. code-block::
      
                  pip install plaidml-keras plaidbench
                  plaidbench keras mobilenet
      
              You can adapt any Keras code by using the PlaidML backend instead of the TensorFlow, CNTK, or Theano backend that you'd normally use; simply change the Keras backend to ``plaidml.keras.backend``. 
              You can do this by modifying
      
              ``~/.keras/keras.json`` so that the backend line reads ``"backend":
              "plaidml.keras.backend"`` If this file does not exist, see the [Backend
              instructions for Keras]. If you don't need anything special in your Keras
              settings, you can set the ``~/.keras/keras.json`` as follows:
      
              .. code-block:: 
                  
                  {
                      "epsilon": 1e-07,
                      "floatx": "float32",
                      "image_data_format": "channels_last",
                      "backend": "plaidml.keras.backend"
                  }
      
              Another option is to globally set the ``KERAS_BACKEND`` environment variable
              to `plaidml.keras.backend`.
              A monkey-patch technique involving ``plaidml.keras.install_backend()`` may still
              work, but should be considered deprecated in favor of the above methods.
      
      
          .. tab:: macOS
      
              A computer listed on `Apple's compatibility list <https://support.apple.com/en-us/HT202823>`_ with support for OpenCL 1.2 is
              required; those from 2011 and later usually fit this requirement.
      
              **Python**
              
              Although PlaidML can be run with Python2, we recommend Python3, as well as
              judicious use of a `Virtualenv <https://virtualenv.pypa.io/en/stable>`_.  To create one just for using PlaidML:
      
              .. code-block:: 
      
                  python3 -m venv plaidml-venv
                  source plaidml-venv/bin/activate
      
              **Keras**
              
              To install PlaidML with Keras, run the following:
      
              .. code-block::
      
                  pip install -U plaidml-keras
      
              Finally, set up PlaidML to use a preferred computing device:
      
              .. code-block::
      
                  plaidml-setup
      
              PlaidML should now be installed! You can test the installation by running
              MobileNet in `plaidbench <https://github.com/plaidml/plaidml/tree/plaidml-v1/plaidbench>`_.
      
              .. code-block::
      
                  pip install plaidml-keras plaidbench
                  plaidbench keras mobilenet
      
          .. tab:: Microsoft Windows
      
              These instructions assume Windows 10 without Python installed; adapt accordingly.
      
              1. First install `Chocolatey <https://chocolatey.org/>`_ by starting an Administrator PowerShell and running:
      
              .. code-block::
      
                  Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
      
              You'll likely need to reboot your shell at this point.
      
              2. Install Python:
      
              .. code-block::
      
                  choco install -y python git vcredist2015
      
              3. Switch to an unprivileged PowerShell to install and set up PlaidML with Keras
      
              .. code-block:: shell
      
                  pip install -U plaidml-keras
                  plaidml-setup
      
              PlaidML should now be installed! You can test the installation by running
              MobileNet in `plaidbench <https://github.com/plaidml/plaidml/tree/plaidml-v1/plaidbench>`_.
      
              .. code-block:: shell
      
                  pip install plaidml-keras plaidbench
                  plaidbench keras mobilenet
      
      
      
      `Intel® SDK for OpenCL™ Applications <https://software.intel.com/en-us/intel-opencl>`_

   .. group-tab:: Building From Source
      Building from source
      ####################
      Install Anaconda
      ==================
      Install `Anaconda <https://www.anaconda.com/download>`_.  You'll want to use a Python 3 version.
      After installing Anaconda, you'll need to restart your shell, to pick up its
      environment variable modifications (i.e. the path to the conda tool and shell
      integrations).
      For Microsoft Windows, you'll also need the Visual C++ compiler (2017+) and the
      Windows SDK, following the `Bazel-on-Windows <https://docs.bazel.build/versions/master/windows.html>`_ instructions.
      
      Install bazelisk
      ==================
      The `Bazelisk <https://github.com/bazelbuild/bazelisk>`_ tool is a wrapper for `Bazel <http://bazel.build>`_ which provides the ability to
      enfore a particular version of Bazel. 
      Download the latest version for your platform and place the executable somewhere
      in your PATH (e.g. ``/usr/local/bin``). You will also need to mark it as
      executable. Example:
      
      .. code-block::
          
          wget https://github.com/bazelbuild/bazelisk/releases/download/v0.0.8/bazelisk-darwin-amd64
          mv bazelisk-darwin-amd64 /usr/local/bin/bazelisk
          chmod +x /usr/local/bin/bazelisk
      
      https://github.com/bazelbuild/bazelisk/releases
      
      Configure the build
      =====================
      Use the `configure` script to configure your build. Note: the `configure` script
      requires Python 3.
      By default, running the `configure` script will:
      * Create and/or update your conda environment
      * Configure pre-commit hooks for development purposes
      * Configure bazelisk based on your host OS
      
      .. code-block::
          
          ./configure
      
      On Windows, use:
      
      .. code-block::
          
          python configure
      
      Here's an example session:
      
      .. code-block::
      
          $ ./configure
          Configuring PlaidML build environment
          conda found at: /usr/local/miniconda3/bin/conda
          Creating conda environment from: $HOME/src/plaidml/environment.yml
          Searching for pre-commit in: $HOME/src/plaidml/.cenv/bin
          pre-commit installed at .git/hooks/pre-commit
          bazelisk version
          Bazelisk version: v0.0.8
          Starting local Bazel server and connecting to it...
          Build label: 0.28.1
          Build target: bazel-out/darwin-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
          Build time: Fri Jul 19 15:22:50 2019 (1563549770)
          Build timestamp: 1563549770
          Build timestamp as int: 1563549770
          Using variant: macos*x86*64
          Your build is configured.
          Use the following to run all unit tests:
          bazelisk test //...
      
      Build the PlaidML Python wheel
      ==============================
      
      .. code-block::
          
          bazelisk build //plaidml:wheel
      
      Install the PlaidML Python wheel
      ==================================
      
      .. code-block::
      
          pip install -U bazel-bin/plaidml/wheel.pkg/dist/*.whl
          plaidml-setup
      
      PlaidML with Keras
      ==================
      The PlaidML-Keras Python Wheel contains the code needed for
      integration with Keras.
      You can get the latest release of the PlaidML-Keras Python Wheel by
      running:
      
      .. code-block::
      
          pip install plaidml-keras
      
      You can also build and install the wheel from source.
      
      Set up a build environment
      ==========================
      Follow the setup instructions for :ref:`Build the PlaidML Python wheel`, above.
      
      Build the PlaidML-Keras wheel
      ===============================
      
      .. code-block::
          
          bazelisk build //plaidml/keras:wheel
      
      Install the PlaidML-Keras Python wheel
      ======================================
      
      .. code-block::
      
          pip install -U bazel-bin/plaidml/keras/wheel.pkg/dist/*.whl
      
      Testing PlaidML
      ===============
      Unit tests are executed through bazel:
      
      .. code-block::
      
          bazelisk test //...
      Unit tests for frontends are marked manual and must be executed individually (requires
      running ``plaidml-setup`` prior to execution)
      
      .. code-block::
          
          bazelisk run //plaidml/keras:backend_test

   .. group-tab:: eDSL Developer Environment

      // TODO: add instructions here
