package dev.g8.neuralnet.network.impl;

import dev.g8.neuralnet.layers.api.AbstractLayer;
import dev.g8.neuralnet.layers.impl.DenseHiddenLayer;
import dev.g8.neuralnet.layers.impl.InputLayer;
import dev.g8.neuralnet.layers.impl.OutputLayer;
import dev.g8.neuralnet.network.api.AbstractNetwork;
import dev.g8.neuralnet.utils.objects.DataSet;

import java.util.*;

/**
 * @author G8LOL
 * @since 4/5/2023
 */
public final class FeedForwardNeuralNetwork extends AbstractNetwork {

	private final Random random = new Random();

	public final void setup() {
		//set prev and next layers
		for (int i = 0; i < layers.size(); i++) {
			//skip input layer
			if (i > 0)
				layers.get(i).setPrevLayer(layers.get(i - 1));

			//skip output layer
			if (i < layers.size() - 1)
				layers.get(i).setNextLayer(layers.get(i + 1));
		}
	}

	@Override
	public final void train(final DataSet dataSet, final int epochs, final int batchSize) {
		//will be used for optimization algorithms
		switch (optimizationAlgorithm) {
			case STOCHASTIC_GRADIENT_DESCENT -> {
				for (int i = 0; i < epochs; i++) {
					System.out.println("epoch: " + i + " error: " + getError(dataSet));

					//shuffle data
					dataSet.shuffle();

					for (int j = 0; j < dataSet.size(); j++) {
						//get random data
						final int randomIndex = random.nextInt(dataSet.size());

						iterate(dataSet.getInput(randomIndex), dataSet.getOutput(randomIndex));
					}
				}
			}
			case MINI_BATCH_GRADIENT_DESCENT -> {
				for (int i = 0; i < epochs; i++) {
					System.out.println("epoch: " + i + " error: " + getError(dataSet));

					//shuffle data
					dataSet.shuffle();

					//batch training
					for (int j = 0; j < dataSet.size(); j += batchSize) {
						//get batch
						final double[][] batchInput = new double[batchSize][];
						final double[][] batchOutput = new double[batchSize][];

						for (int k = 0; k < batchSize; k++) {
							batchInput[k] = dataSet.getInput(j + k);
							batchOutput[k] = dataSet.getOutput(j + k);
						}

						for (int k = 0; k < batchSize; k++) {
							iterate(batchInput[k], batchOutput[k]);
						}
					}
				}
			}
			case GRADIENT_DESCENT -> {
				for (int i = 0; i < epochs; i++) {
					System.out.println("epoch: " + i + " error: " + getError(dataSet));

					for (int j = 0; j < dataSet.size(); j++)
						iterate(dataSet.getInput(j), dataSet.getOutput(j));
				}
			}
		}
	}

	@Override
	public final void iterate(final double[] input, final double[] output) {
		//forward propagation
		double[] lastLayerOutput = input;
		double[][] lastLayerWeights = null;

		for (AbstractLayer layer : layers) {
			layer.computeForward(lastLayerOutput, lastLayerWeights);

			lastLayerOutput = layer.getOutput();
			lastLayerWeights = layer.getWeights();
		}

		//backward propagation

		//basically backpropagate the error from the output layer to the hidden layer

		/*
		 * output layer weights delta = error * output layer activation function derivative * hidden layer output
		 * hidden layer weights delta = sum of (error * output layer weights) * hidden layer activation function derivative * input layer output
		 *
		 * update weights = weights - learning rate * weights delta
		 *
		 * learning rate = 0.1
		 *
		 * also weights of a layer are the weights between the current layer and next layer
		 *
		 * so if a layer was setup like this:
		 *
		 * input layer -> hidden layer -> output layer
		 *
		 * then the weights of the input layer are the weights between the input layer and the hidden layer
		 *
		 * might change this later
		 */

		//reverse layers
		List<AbstractLayer> reversedLayers = new ArrayList<>(layers);

		Collections.reverse(reversedLayers);

		for (final AbstractLayer layer : reversedLayers) {
			if (layer instanceof InputLayer)
				continue;

			if (layer instanceof final OutputLayer outLayer) {
				//compute error of output layer
				final double[] outputErrors = outLayer.computeBackprop(layer.getOutput(), output);

				//update weights/biases
				outLayer.updateWeights((DenseHiddenLayer) outLayer.getPrevLayer(), outputErrors, outLayer.getPrevLayer().getOutput(), learningRate);
				outLayer.updateBiases(outputErrors, learningRate);
			} else if (layer instanceof final DenseHiddenLayer hidLayer){
				//AbstractLayer nextLayer = reversedLayers.get(reversedLayers.indexOf(layer) - 1);

				//compute error of hidden layer
				final double[] hiddenErrors = hidLayer
						.computeBackprop(hidLayer.getNextLayer(), hidLayer.getOutput(), hidLayer.getNextLayer().getOutputErrors());

				//update weights/biases
				hidLayer.updateWeights(hidLayer.getPrevLayer(), hiddenErrors, layer.getPrevLayer().getOutput(), learningRate);
				hidLayer.updateBiases(hiddenErrors, learningRate);
			}
		}
	}

	@Override
	public final DataSet predict(final DataSet input) {
		final double[][] output = new double[input.size()][input.getInput(0).length];

		//forward propagation
		for (int i = 0; i < input.size(); i++) {
			double[] lastLayerOutput = input.getInput(i);
			double[][] lastLayerWeights = null;

			for (AbstractLayer layer : layers) {
				layer.computeForward(lastLayerOutput, lastLayerWeights);

				lastLayerOutput = layer.getOutput();
				lastLayerWeights = layer.getWeights();
			}

			//set the respective output
			output[i] = lastLayerOutput;
		}

		return new DataSet(input.getInputsArray(), output);
	}

	/**
	 * predict the output of the network using just one input
	 * @param input the input
	 * @return the output
	 */
	private final double[] predict(final double[] input) {
		//forward propagation
		double[] lastLayerOutput = input;
		double[][] lastLayerWeights = null;

		for (AbstractLayer layer : layers) {
			layer.computeForward(lastLayerOutput, lastLayerWeights);

			lastLayerOutput = layer.getOutput();
			lastLayerWeights = layer.getWeights();
		}

		return lastLayerOutput;
	}

	/**
	 * get the error of the network using MSE owo
	 * @param dataSet
	 * @return the error
	 */
	private final double getError(final DataSet dataSet) {
		double error = 0;

		for (int i = 0; i < dataSet.size(); i++) {
			final double[] output = predict(dataSet.getInput(i));

			for (int j = 0; j < output.length; j++) {
				error += Math.pow(dataSet.getOutput(i)[j] - output[j], 2);
			}
		}

		return error;
	}

}
