# Noesis-Py: TensorFlow graph construction tool-set

## Install

First we must install virtualenv:
```bash
pip install virtualenv
```

Then we create a virtualenv for Python3 and activate it:
```bash
virtualenv ~/.virtualenvs/noesis -p /usr/bin/python3.5
source ~/.virtualenvs/noesis/bin/activate
```

Before installing any Python packages, first we must upgrade PIP:
```bash
pip install -U pip
```

If your machine does not have an NVIDIA graphics card, you can install the CPU-ony version:
```bash
cd $NOESIS_ROOT/noesis_py
pip install -e .[cpu]
```

And if you do have an NVIDIA GPU, then you can run:
```bash
cd $NOESIS_ROOT/noesis_py
pip install -e .[gpu]
```

## Examples

There is a very simple example of how to generate a Noesis computation-graph using the `noesis_py` API in `examples/`. You can 
experiment with generating different graphs by modifying the `description.xml`.

First lets change the architecture of the Multi-Layer Perceptron (MLP) neural-network which we use as a function-approximation to represent 
the policy. The latter is of type `DiagonalGaussianPolicy` which just outputs the mean and a diagonal covariance matrix directly
. We will set the number of layers and the units per-layer as well as the non-linearity in each layer:
```XML
<TODO/>
```

Now we can generate a graph based on this description by just running the script:
```bash
cd $NOESIS_ROOT/noesis_py/examples
./graph_generation_from_xml.py
```

This script will automatically launch TensorBoard in the end, so you can use the link to open a window in Chrome and inspect the generated
graph for yourself.

## Unit Tests

For unt-tests we use `pytest`. In order to execute the tests you may run the following:
```bash
cd $NOESIS_ROOT/noesis_py
pytest
```


## PyLint

If you want to run the Python linter and inspect PEP8 code compliance, you can run:
```bash
cd $NOESIS_ROOT/noesis_py
pylint noesis
```


----

