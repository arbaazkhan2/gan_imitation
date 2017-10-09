# InfoGAN

Code for reproducing key results in the paper [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel.

## Dependencies

This fork of [the original](https://github.com/openai/InfoGAN) changes part of the code to accomodate the new TensorFlow API.  I am using version 1.3.0.  Since this project also used `prettytensor`, which has it's own problems, you'll need to make a fix to this library as well (instructions below). 

In addition, please `pip install` the following packages:
- `prettytensor`
- `progressbar2`
- `python-dateutil`

### Modifying PrettyTensor
You'll likely get an error like `ValueError:  too many values to unpack`.  You can fix this by either pulling the relevant commit from the pull request or merely making [this](https://github.com/google/prettytensor/pull/57/files) trivial change.  


## Running Experiment

We provide the source code to run the MNIST example:

```bash
PYTHONPATH='.' python launchers/run_mnist_exp.py
```

You can launch TensorBoard to view the generated images:

```bash
tensorboard --logdir logs/mnist
```
