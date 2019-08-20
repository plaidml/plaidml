# Transfer Learning Demo

## Getting Started
Use of a virtual enviornment is highly recommended although not required.
```bash
virtualenv .venv
source .venv/bin/activate
```

`pip install -r requirements.txt` will install the following packages necessary for this demo to run.
- ngraph_tensorflow_bridge
- tensorflow==1.14.0
- plaidml-keras
- pillow
- matplotlib
- jupyter

## Other details
Menu Bar --> Cell --> Current Outputs --> Toggle Scrolling to stop Jupyter Notebook from creating scrollable boxes when the images appear.

Instantiation of `TransferLearningDemo` with `verbose=1` will create a third output box where training and testing functions will output their loss and accuracy figures, along with how long the training took.
```python
d = TransferLearningDemo.Demo(verbose=1)
```
