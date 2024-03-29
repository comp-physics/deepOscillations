## Neural networks can be FLOP-efficient integrators of 1D oscillatory integrands

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

> We demonstrate that neural networks can be FLOP-efficient integrators of one-dimensional oscillatory integrands. We train a feed-forward neural network to compute integrals of highly oscillatory 1D functions. The training set is a parametric combination of functions with varying characters and oscillatory behavior degrees. Numerical examples show that these networks are FLOP-efficient for sufficiently oscillatory integrands with an average FLOP gain of 1000 FLOPs. The network calculates oscillatory integrals better than traditional quadrature methods under the same computational budget or number of floating point operations. We find that feed-forward networks of 5 hidden layers are satisfactory for a relative accuracy of 0.001. The computational burden of inference of the neural network is relatively small, even compared to inner-product pattern quadrature rules. We postulate that our result follows from learning latent patterns in the oscillatory integrands that are otherwise opaque to traditional numerical integrators.

![Project Image or GIF](https://github.com/comp-physics/deepOscillations/blob/master/doc/NN_integral.gif)

### Dependencies

Before you begin, ensure you have met the following requirements:

- DeepXDE (`pip install deepxde`)
- Tensorflow 3.8 or higher (`pip install tensorflow`)

### Installation

To reproduce our results, follow these steps:
```bash
git clone https://github.com/comp-physics/deepOscillations.git
cd deepOscillations
bash bash_main.sh
```

### License

[MIT](https://opensource.org/licenses/MIT).
