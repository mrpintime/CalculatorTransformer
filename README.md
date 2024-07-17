# CalculatorTransformer

This repository aims to implement an approach to solving basic mathematical calculations using transformer models. The goal is to explore how transformer architectures, typically used for natural language processing tasks, can be adapted to perform arithmetic operations and basic mathematical functions.

## Overview

This project investigates the feasibility of using transformer models to solve basic mathematical calculations. By leveraging the self-attention mechanism and powerful learning capabilities of transformers, we aim to create a model capable of performing arithmetic operations such as addition, subtraction, multiplication, and division.

## Approach

The basic idea is to adapt the transformer structure initially used for translation in the famous paper "Attention Is All You Need". We take a math expression like "2+6" or "10*16", tokenize it, and pass it as input to the encoder part of the transformer. The decoder part then generates the answer, which is tokenized using the same tokenizer and passed to the decoder. This approach operates similarly to a translation transformer.

During the training phase, the model uses the following inputs and outputs:

**Inputs:**
- encoder_input
- encoder_mask
- decoder_input
- decoder_mask

**Outputs:**
- encoder_output
- decoder_output
- proj_output: projection of decoder output into the token space (vocabulary space)

## Usage

You can train model by running `main.py` with following setup for `train_model` function:  

```python
train_model(config, val=False) # for specifying epoch for training you can look at Configs.py file
```

If you only want evaluate your model on validation set, You can run `main.py` with following setup for `train_model` function:  

```python
train_model(config, val=True, num_example=10) # num_example= number of example you want to test 
```


## Results

The results with my dataset and tokenizer were not satisfactory. However, you can create your own dataset and tokenizer using the same approach to see what improvements can be achieved.

## Acknowledgments

- This project was inspired by [pytorch-transformer](https://github.com/hkproj/pytorch-transformer).
