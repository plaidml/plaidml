To add new toy networks (along the lines of toy_parallel_conv), you need to make several changes:
 - Define your network in `plaidbench/plaidbench/networks/keras/<name>.py`, where you need to implement two functions:
   - `build_model`, which takes no parameters and returns a `keras.models.Model`
   - `scale_dataset`, which takes one parameter (which will be a numpy tensor of shape Nx32x32x3 containing N inputs); you are to reshape the non-batch dimensions of this input to produce an appropriate input shape for your model. If reshaped images like this are not appropriate for your use case, see `imdb_lstm` for an example of the (more involved) code needed to support other inputs.
 - Add this file to the list in `plaidbench/CMakeLists.txt`
 - Add your network to the legal networks lists in `plaidbench/plaidbench/__init__.py` and `plaidbench/plaidbench/frontend_keras.py`

If you want to enable correctness tests, you will need to take a few additional steps. But first, a warning: If you set up tests with random weights, the compounding floating point errors from reordering computations can be considerably larger than when running a trained network. These often are larger than the plaidbench error tolerances (indeed, they are larger than the tolerances for `toy_parallel_conv`) -- you will need to judge whether these are just tolerance errors as I just described or if there is a more fundamental correctness problem. (Adding correctness tests with trained weights is more robust, but traing is a substantially more complex process that isn't covered in this document.)

To set up the correctness tests, you will need to use options in both sections of the plaidbench CLI: In general, plaidbench is run for Keras as `plaidbench <global-options> keras <keras-options> <model-name>`

To add correctness tests:
 - Add a `model.save_weights` call to the end of the `build_model` function, analogous to the `load_weights` call in `toy_parallel_conv`
 - Run the model using the backend you want to test against, with the `--results` parameter (a global option) pointing to the directory where you want the model output saved. Also be sure to use the batch size you want to test against (probably 1)
   - If you want to run with tensorflow to get a non-PlaidML set of weights to test against, pass `--tensorflow` as a keras-specific option. Also be sure that in your `.keras/keras.json` config file that the backend is set to `tensorflow`.
 - Save the `.npy` results file as `plaidbench/golden/<network-name>/infer,bs-1.npy` (assuming these are batch size 1 results, otherwise adjust the number accordingly)
 - Save the `.h5` weights file as `plaidbench/networks/keras/<network-name>.h5`
 - Add these two files to the `plaidbench/CMakeLists.txt` file list
 - Replace the `save_weights` call in `build_model` with a `load_weights` call
