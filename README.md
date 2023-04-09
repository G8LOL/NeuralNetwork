# Neural Network

A framework for neural networks written entirely in Java. Contains an implementation of a multi-perceptron feed forward neural network (FFNN). Can be used for tasks such as classification, regression, and prediction (example XOR problem used for the FFNN).

![neural network](https://user-images.githubusercontent.com/66517997/230747863-c275fc71-dff6-4069-8d75-b81e76491d79.png)

## How to use ##
Clone the repo with Intellij or your IDE of choice. Make sure to use Java 17. **Main.java** contains the network architecture and the example XOR problem.

You can customize the architecture of the neural network further. There are a few optimization algorithms (gradient descent, stochastic gradient descent, etc), several activation and loss functions, different weight initializations. You can also add new layers or change the learning rate (default is 0.1).

## How it works ##
The neural network implemented in this project is a feed forward neural network where the data flows through the layers in one direction. Each layer consists of multiple neurons, which receive inputs from the previous layer and produce outputs that are fed into the next layer. 

The formula for calculating the weighted sum for a neuron:

`weighted_sum = (input_1 * weight_1) + (input_2 * weight_2) + ... + (input_n * weight_n)`

You then apply an activation function (e.g logistic sigmoid which maps inputs to values between 0 and 1) used to introduce non-linearity, and the ability to learn more complex patterns, to the network.

logistic sigmoid formula:

`1 / (1 + e^(-x))`

The network is then trained using **back propagation** which basically just adjusts the weights and biases of the neurons in order to minimize the error between the actual output and the desired output. During this process, the errors are propagated through the layers in reverse, from the output to the input layer. To do this, we use have to use the chain rule of calculus to calculate the gradient of the error with respect to each weight and bias in the network. This gradient tells us how much we should adjust each weight and bias to reduce the error.

The gradient/delta is calculated using the following formula which you can also see in the code:

`delta = error * derivative_of_activation_function(output) * input`

[In Depth Explanation of Gradient Descent](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931)

[Good Video on the Math Behind Backprop](https://www.youtube.com/watch?v=tIeHLnjs5U8)

## TODO/For Later
- Add Convolutional Layers
- Add Recurrent Layers
- Dropout Layer

## License
This project is licensed under the MIT License (See LICENSE file for more info)
