# deepOscillations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

Evaluating highly-oscillatory integrals via deep operator networks

> This work demonstrates the existence of FLOP-efficient integrators for one-dimensional oscillatory integrals. The simple feed-forward architecture efficiently calculates oscillatory integrals with high precision and low computational cost compared to current numerical integration methods. The presented method holds applicability where numerical integrals of one-dimensional oscillating functions are required either as an end-to-end problem or a sub-problem.

![Project Image or GIF](https://github.com/comp-physics/deepOscillations/blob/master/doc/NN_integral.gif)

## Dependencies

Before you begin, ensure you have met the following requirements:

- DeepXDE (`pip install deepxde`)
- Tensorflow 3.8 or higher (`pip install tensorflow`)

## Installation

To reproduce our results, follow these steps:
```bash
git clone https://github.com/comp-physics/deepOscillations.git
cd deepOscillations
bash bash_main.sh
```

## License

This project is licensed under the terms of the [MIT license](https://opensource.org/licenses/MIT).
