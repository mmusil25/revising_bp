## Test Helper CLI Tools

### Requirements

* Python Version >= 3.5
* `numpy`
* `sklearn`  (only for `train_test_split`)
* `scipy` (for sparse singular solver)
* `matplotlib`

### Basic Usage


#### Run tests on multiple datasets. (Recommended method) 

Use `[Python Exec] run_all_tests.py` to run all the test configure files (`.json`) under `./test_config/` directory which is a sub-directory of the current directory.
To configure the `.json` files you need choose the arguments as detailed in the 'Args' section below. 


#### Run a single test as configured in the 'config.json' file found in this directory.   
`[Python Exec] mlp_tests_gen.py -R`
Example: (from Linux terminal) >>python mlp_tests_gen.py -R


##### Args

*  `--dataset`: *Required. [String].* Choose the dataset to train the model.  Should be one of following option: `{winequality,iris,digits,boston,diabetes,covtype,california_housing,olivetti_faces,mnist}`
* `--testratio`: *Optional. [Float (0 - 1)]*. Set the train test split ratio. (Default `0.1`)
* `--shuffle`: *Optional. [Boolean]*. Whether shuffle the training set for each epoch. (Default `True`)
* `--epochs`: *Optional. [Integer]*. Number of epochs. (Default `100`)
* `--optimizer`: *Required. [String]*. Set the algorithm of optimizer. Should be one of following choices: `{sgd, layer, neuron}`.
* `--lr`: *Optional. [Float].* Learning Rate. (Default `0.01`)
* `--hdims`: *Required. [List of Integers]*. Set the hidden dimensions. E.g. `3 4 5` means the multilayer perceptron has `3 4 5 output_dim` neurons for each layer . Format: use space to split integers.
* `--layer-coeff`: *[List of Float].* If the optimizer is `layer`, this parameter is required. Suggest to set lower coefficient for the first layers, and higher coefficient in the end. The length of the list should be `length(hdims) + 1`.  (Default `None`)
* `--batch-size`: *Optional. [Integer]*. Set the batch size for training. (Default `50`)


#### Generate batch file
`[Python Exec] mlp_tests_gen.py`

----

----

### Helper Script - `mlp_tester`

This script is used for setting up the training and testing parameters for the model in command line.

#### Usage

`[Python Exec] mlp_tester.py [--ARGSKEY ARGSVALUE]`

##### Python Exec

For `Windows` users, this term should be the executable file if you didn't set the global variable.

For `Linux` and `macOS` users, this term should be `python3`. 


#### Examples

```bash
python3 mlp_tester.py --dataset iris --testratio 0.1 --lr 0.1 --optimizer sgd --epochs 200 --hdims 10 6 4

python3 mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.1 --optimizer layer --epochs 100 --hdims 100 50 25 --layer-coeff 0.3, 0.5, 1, 2
```



----

### Helper Scripts - `mlp_tests_gen`

This script is used for generating a sort of test cases based on the `JSON` configuration file. By default, it will generate all the test cases by using `mlp_tester.py` and store the commands in `.sh` or `.bat` file (based on detecting the users' platform).

#### Usage

`[Python Exec] mlp_test_gen.py [-OPTION]|[--ARGSKEY ARGSVALUE]`

##### Args

* `-R` or `--run`. If this option is set, it will run all test cases directly, rather than generating the batch file.
* `--pythonpath`. *Optional. [String].* Set the Python executable file or command. (Default: `None`)
* `--testscript`. *Optional. [String].* Set the path of test script. (Default: `mlp_tester.py`)
* `--config`. *Optional. [String].* Set the JSON configuration file path. (Default: `config.json`)

##### Examples

```bash
python3 mlp_tests_gen.py -R --config ./test_config/some_configs.json
```

#### JSON Configuration

`mlp_tests_gen.py` will iterate all the possible combinations for all the elements in `JSON` file. The configurations in `JSON` file should be in following format.

##### Example

The following `JSON` file is placed in `./config.json` as a template.

```json
{
    "learning_rates": [0.1, 0.01],
    "optimizers": ["sgd", "layer", "neuron"],
    "datasets": ["mnist"],
    "testratio": [0.1],
    "epochs": [100],
    "hdims": [
        [100, 60, 40],
        [80, 40, 30]
    ],
    "layer_coeff": [
        [0.3, 0.5, 1, 2]
    ]
}
```

Applying `mlp_tests_gen.py` on this `JSON` file (i.e. `python3 mlp_tests_gen.py  --config ./config.json`) will generate the following commands (at `./mlp_test.bat` or `./mlp_test.sh`):

```bash
python mlp_tester.py --dataset mnist --testratio 1.1 --lr 0.1 --optimizer sgd --epochs 100 --hdims 100 60 40
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.1 --optimizer sgd --epochs 100 --hdims 80 40 30
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.1 --optimizer layer --epochs 100 --hdims 100 60 40 --layer-coeff 0.3 0.5 1 2
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.1 --optimizer layer --epochs 100 --hdims 80 40 30 --layer-coeff 0.3 0.5 1 2
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.1 --optimizer neuron --epochs 100 --hdims 100 60 40
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.1 --optimizer neuron --epochs 100 --hdims 80 40 30
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.01 --optimizer sgd --epochs 100 --hdims 100 60 40
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.01 --optimizer sgd --epochs 100 --hdims 80 40 30
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.01 --optimizer layer --epochs 100 --hdims 100 60 40 --layer-coeff 0.3 0.5 1 2
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.01 --optimizer layer --epochs 100 --hdims 80 40 30 --layer-coeff 0.3 0.5 1 2
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.01 --optimizer neuron --epochs 100 --hdims 100 60 40
python mlp_tester.py --dataset mnist --testratio 0.1 --lr 0.01 --optimizer neuron --epochs 100 --hdims 80 40 30
```



----

### Helper Scripts - `run_all_tests.py`

To test the model by using all the configuration files under `./test_config/`.

#### Usage

`[Python Exec] run_all_tests.py`

----

### More DOCS

Use `[command] --help` to get help docs for CLI tools.
