# Generating Handwritten Digits Using GANs

## About
This project implements a Generative Adversarial Network (GAN) to generate handwritten digit images similar to those in the MNIST dataset. 
The GAN consists of two neural networks, a Generator and a Discriminator, which are trained simultaneously by competing against each other.

Generative Adversarial Networks (GANs) are a class of machine learning frameworks designed to generate new data samples that resemble the training data. 
In this project, a GAN is trained to generate images of handwritten digits (0-9) similar to those found in the MNIST dataset.

## Requirements

The project requires the following dependencies:

- Python 3.x
- `numpy`
- `matplotlib`
- `scikit-learn`


## Example

### Generating Handwritten Digits
    $ python mlfromscratch/unsupervised_learning/generative_adversarial_network.py

    +-----------+
    | Generator |
    +-----------+
    Input Shape: (100,)
    +------------------------+------------+--------------+
    | Layer Type             | Parameters | Output Shape |
    +------------------------+------------+--------------+
    | Dense                  | 25856      | (256,)       |
    | Activation (LeakyReLU) | 0          | (256,)       |
    | BatchNormalization     | 512        | (256,)       |
    | Dense                  | 131584     | (512,)       |
    | Activation (LeakyReLU) | 0          | (512,)       |
    | BatchNormalization     | 1024       | (512,)       |
    | Dense                  | 525312     | (1024,)      |
    | Activation (LeakyReLU) | 0          | (1024,)      |
    | BatchNormalization     | 2048       | (1024,)      |
    | Dense                  | 803600     | (784,)       |
    | Activation (TanH)      | 0          | (784,)       |
    +------------------------+------------+--------------+
    Total Parameters: 1489936

    +---------------+
    | Discriminator |
    +---------------+
    Input Shape: (784,)
    +------------------------+------------+--------------+
    | Layer Type             | Parameters | Output Shape |
    +------------------------+------------+--------------+
    | Dense                  | 401920     | (512,)       |
    | Activation (LeakyReLU) | 0          | (512,)       |
    | Dropout                | 0          | (512,)       |
    | Dense                  | 131328     | (256,)       |
    | Activation (LeakyReLU) | 0          | (256,)       |
    | Dropout                | 0          | (256,)       |
    | Dense                  | 514        | (2,)         |
    | Activation (Softmax)   | 0          | (2,)         |
    +------------------------+------------+--------------+
    Total Parameters: 533762


<p align="center">
    <img src="gan_mnist5.gif" width="640">
</p>
<p align="center">
    Figure: Training progress of a Generative Adversarial Network generating <br>
    handwritten digits.
</p>