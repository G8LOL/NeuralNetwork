package dev.g8.neuralnet;

import dev.g8.neuralnet.functions.activation.ActivationFunction;
import dev.g8.neuralnet.functions.loss.LossFunction;
import dev.g8.neuralnet.initializations.WeightInitialization;
import dev.g8.neuralnet.layers.impl.DenseHiddenLayer;
import dev.g8.neuralnet.layers.impl.DropoutLayer;
import dev.g8.neuralnet.layers.impl.InputLayer;
import dev.g8.neuralnet.layers.impl.OutputLayer;
import dev.g8.neuralnet.network.api.AbstractNetwork;
import dev.g8.neuralnet.network.impl.FeedForwardNeuralNetwork;
import dev.g8.neuralnet.optimizations.OptimizationAlgorithm;
import dev.g8.neuralnet.utils.objects.DataSet;

public class Main {
	
	public static void main(String[] args) {
		final AbstractNetwork feedForwardNeuralNetwork = new FeedForwardNeuralNetwork.NetworkBuilder(FeedForwardNeuralNetwork.class)
				.withLearningRate(0.1)
				.withOptimizationAlgorithm(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.withLayers(
						new InputLayer(2, 4,
								WeightInitialization.XAVIER),
						new DenseHiddenLayer(4, 4,
								ActivationFunction.TANH,
								WeightInitialization.XAVIER),
						//new DropoutLayer(4, 0.1),
						new DenseHiddenLayer(4, 1,
								ActivationFunction.TANH,
								WeightInitialization.XAVIER),
						//new DropoutLayer(4, 0.3),
						new OutputLayer(1, 1,
								ActivationFunction.LOGISTIC_SIGMOID,
								LossFunction.MEAN_SQUARED_ERROR,
								WeightInitialization.XAVIER)
				)
				.build();

		//xor problem
		final double[][] input = {
				{0, 0},
				{0, 1},
				{1, 0},
				{1, 1}
		};

		final double[][] output = {
				{0}, {1}, {1}, {0}
		};

		final DataSet dataSet = new DataSet();

		dataSet.add(input, output);

		if (!(feedForwardNeuralNetwork instanceof FeedForwardNeuralNetwork)) {
			System.out.println("not a feed forward neural network");
			return;
		}

		feedForwardNeuralNetwork.printNetwork();

		System.out.println("training...");

		feedForwardNeuralNetwork.setup();

		feedForwardNeuralNetwork.train(dataSet, 10000, 2);

		System.out.println("testing...");

		final DataSet prediction = feedForwardNeuralNetwork.predict(dataSet.getInputs());

		//print results
		prediction.print();
	}

}
