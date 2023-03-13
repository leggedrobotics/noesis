# Noesis Python

## Install

First we must install virtualenv:
```bash
pip install virtualenv
```

Then we create a virtualenv for Python3 and activate it:
```bash
virtualenv ~/.virtualenvs/noesis -p /usr/bin/python3.7
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

----
