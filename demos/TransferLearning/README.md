## Transfer Learning Demo

## Getting Started
Use of a virtual enviornment is highly recommended although not required. 
```bash
virtualenv .venv
source .venv/bin/activate
```

`pip install -r requirements.txt` will install the following packages necessary 
for this demo to run.
- ngraph_tensorflow_bridge
- tensorflow==1.14.0
- plaidml-keras
- pillow
- matplotlib
- jupyter

For PlaidML devices, run `plaidml-setup` before starting this demo to select the
Plaidml backend device.

Once the interface is presented, setup the model and network along with epoch 
and batch size values then hit the Train button to start. The demo will 
initialize test images and then start with a prediction with the model selected 
using pretrained `imagenet` weights. The demo will then automatically proceed 
into the training phase followed by another predition phase based on the trained
model.

Initially, the demo will select 9 random images from the test set and perform 
inference on them using the pre-trained model. These initial guesses are often 
incorrect. After training, the demo will perform inference on the same 9 images 
using the fully trained model. This is shown to prove that the training phase 
results in improved prediction accuracy.

## Running the demo through the Jupyter Notebook

Open the Jupyter Notebook using the following command:

```python
jupyter notebook TransferLearningDemo.ipynb
```

## Running the demo through the command line
```
usage: TransferLearningDemo [-h] [--training] 
                            [--network_type NETWORK_TYPE] [--backend BACKEND]
                            [--warmup] [--workers] [--quiet]

optional arguments:
  -h, --help            show this help message and exit
  --training            performs the training phase of the demo
  --network_type NETWORK_TYPE
                        selects the network used for training/classification
                        [ResNet50]/MobileNet V2
  --backend BACKEND     selects the backend used for training/classification 
                        [CPU]/PLAIDML/TF]
  --warmup              train with a warmup run first
  --workers             identify the number of workers
  --quiet               disables most logging
```
