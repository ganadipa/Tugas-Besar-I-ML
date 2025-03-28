# Tugas-Besar-I-ML


## Setup
Spawn venv, source the venv, then get dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

We tried using Python 3.11 and it didn't work, so install Python 3.12 (or later) if you don't have it yet. 

## Folder Structure

The following is the folder structure.
```
├───doc
└───src
    ├───lib
    ├───models
    └───playground
        ├───dava-tornado
        ├───nyoman-gana
        └───testing
```

You should find:
1. The report in `doc` folder
2. The implementations in `src/lib` 
3. The trained model in `src/models/ffnn_model`
4. The testing `ipynb` files in `src/playground/testing`

## How to Use
Import the class from `src/lib`. Then, to spawn a model, do:

```
model = FFNN(NeuralNetwork(
    node_counts = <Give a list of node counts>,
    activations = <Give a list of activation class instances>,
    loss_function = <Instance of a Loss Function Class>,
    initialize_methods = <Give a list of Weight Initializer Istances> OR a single instance
))
```

OR, if you want to **load** from a file, do:
```
model = FFNN.load('path/to/file')
```

The layer count will be inferred from the `node_counts` length. Here is an example:
```
model = FFNN(NeuralNetwork(
    node_counts = depth_variations[0],
    activations = activation_variations[0],
    loss_function = CCE(),
    initialize_methods = NormalInitializer(seed=22)
))
```

And then to `fit`, call `fit`.

```
model.fit(
    x_train=X_train,
    y_train=y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_val, y_val),
    learning_rate=0.01,
    verbose=1
)
```

Refer to the `testing` files to see how the data is initially processed and converted to `X_train` and `y_train`.

Here are the following other methods per the specification.

1. `predict`
2. `evaluate`
3. `save`
4. `load`
5. `plot_networks`
6. `plot_weights`
7. `plot_loss_history`

Again, refer to the `testing` files if you need an example to any specific method.

## Authors
| **Name**| **NIM**|
|-|-|
|**Renaldy Arief Susanto**|13522022|
|**Nyoman Ganadipa Narayana**|13522066|
|**Muhammad Dava Fathurrahman**|13522114|