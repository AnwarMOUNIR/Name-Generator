# US Name Generator (RNN / GRU)

This repository contains a character-level Recurrent Neural Network (RNN) designed to generate novel, human-sounding names. It is trained on a dataset of US names and uses a Gated Recurrent Unit (GRU) architecture built with TensorFlow and Keras.

## Features

* **Character-Level Prediction:** Learns the spelling patterns of names letter-by-letter to generate brand new ones.
* **Temperature Control:** Implements temperature scaling (T = 1.5) to balance creativity and realistic spelling.
* **Repetition Testing:** Includes a custom benchmarking script to test the model's originality by measuring how many generations it takes before a duplicate name is produced.
* **Memory-Safe Data Slicing:** Efficient sequence shifting to handle the large ~5.6 million row dataset without crashing.

## Requirements

To run this project, you will need Python installed along with the following libraries:

* `pandas`
* `numpy`
* `tensorflow`

## Dataset

The model requires a dataset named `StateNames.csv` located in the root directory. The training script looks specifically for a `Name` column to build its vocabulary and training sequences.

## Usage

### 1. Training the Model

Run the training script to parse the data, build the character vocabulary, and train the neural network. 

```bash
python main.py
```

* **Note on Memory:** The script uses a large batch size (`2048`) to speed up training on the massive dataset. If your machine throws an Out Of Memory (OOM) error, open the training file and lower the batch size to `1024` or `512`.
* **Outputs:** The best performing model during validation is autosaved as `best_name_model.keras`, and the final training weights are saved as `custom_name_generator.keras`.

### 2. Generating Names and Testing

Once you have generated the `.keras` model files, run the testing script.

```bash
python test.py
```

* **Name Generation:** The script will automatically load the learned weights, rebuild the vocabulary from the CSV, and output 20 custom names based on predefined seed letters.
* **Repetition Test:** After generation, it runs a multi-trial collision test. This function measures the average number of generations required before the model repeats a name it has already created.

## Model Architecture

The core of the generator is a Sequential Keras model built with the following layers:

* **Embedding Layer:** Maps characters to 32-dimensional dense vectors, actively masking padded zeros to ignore blank space.
* **GRU Layer:** The "brain" of the model with 128 units, configured to return full sequences to evaluate every letter step.
* **Dense Layer:** Outputs raw predictions (logits) matching the size of the unique character vocabulary.
* **Lambda Layer:** Applies custom temperature scaling to the logits to slow down the softmax confidence.
* **Softmax Activation:** Converts the adjusted logits into an actionable probability distribution for character sampling.
