# README

Answers to questions are submitted in `results.md`.

## Code execution

The Python code for Tasks 1 and 2 are located in `task1.py` and `task2.py`. They behave differently depending on whether we have loaded in model weights `task1.pt` and `task2.pt`. If these weights are not given, then the Python files will train the model on the training set before evaluating them on the test set.

### Setup environment

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Task 1

```
python3 task1.py
```

### Task 2

```
python3 task2.py
```

### Required Files

- `glove.6B.100d.txt`
- `conllevel.py`

### Optional Files

- `./weights/task1.pt`
- `./weights/task2.pt`

### Cleanup

```
deactivate
```
