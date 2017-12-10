# Neural Network Prototype and Iris dataset example

Platform | Build status
---------|-------------:
Linux<br>OSX | [![Build Status](https://travis-ci.org/denizevrenci/neural_network_prototype.svg?branch=master)](https://travis-ci.org/denizevrenci/neural_network_prototype)

## Building and testing
```sh
mkdir build && cd build && cmake .. && cmake --build .
```

## libnnp
libnnp implements a simple feedforward neural network.

Network layers can be formed by making a specialization of the nnp::ComputationalLayer class template.
Multiple layers can be appended with the `nnp::TupleNetwork` class template.
Adding a loss layer to a `nnp::TupleNetwork` and calling the `propagate()` function with the appropriate parameters trains the network a single iteration.
`propagate()` also has an overload to check the loss without back propagation to use with a validation set.
Calling the `forward()` function of `nnp::TupleNetwork` returns the output tensor from the outermost layer. This can be used at test time.

## Iris dataset example
After the project is built, run the program by passing it the path of the iris dataset.

```sh
./build/example/iris/iris_training example/iris/iris.data
```
