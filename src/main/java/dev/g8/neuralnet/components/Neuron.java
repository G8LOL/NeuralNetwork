package dev.g8.neuralnet.components;

import dev.g8.neuralnet.functions.activation.ActivationFunction;
import dev.g8.neuralnet.initializations.WeightInitialization;

/**
 * @author G8LOL
 * @since 3/31/2023
 */
public final class Neuron {

	private double input, output, error, bias;

	private Connection[] connections;

	private boolean inputNeuron = false;

	public Neuron(final int numConnections, final int numOutput, final WeightInitialization weightInitialization) {
		this.output = 0;
		this.error = 0;
		this.bias = Math.random();

		this.connections = new Connection[numOutput];

		for (int i = 0; i < numOutput; i++)
			this.connections[i] = new Connection(weightInitialization.initializeWeight(numConnections, numOutput));
	}

	public Neuron(final int numConnections, final int numOutput, final WeightInitialization weightInitialization, final boolean inputNeuron) {
		this(numConnections, numOutput, weightInitialization);

		this.inputNeuron = inputNeuron;
	}

	/**
	 * get input value
	 * @return
	 */
	public final double getInput() {
		return input;
	}

	/**
	 * set input value
	 * @param input
	 */
	public final void setInput(final double input) {
		this.input = input;
	}

	/**
	 * get output value
	 * @return
	 */
	public final double getOutput() {
		return inputNeuron ? input : output;
	}

	/**
	 * set output value
	 * @param output
	 */
	public final void setOutput(final double output) {
		this.output = output;
	}

	/**
	 * get error value
	 * @return
	 */
	public final double getError() {
		return error;
	}

	/**
	 * set error value
	 * @param error
	 */
	public final void setError(final double error) {
		this.error = error;
	}

	/**
	 * get bias value
	 * @return
	 */
	public final double getBias() {
		return bias;
	}

	/**
	 * set bias value
	 * @param bias
	 */
	public final void setBias(final double bias) {
		this.bias = bias;
	}

	/**
	 * get connections
	 * @return
	 */
	public final Connection[] getConnections() {
		return connections;
	}

	/**
	 * set connections
	 * @param connections
	 */
	public final void setConnections(final Connection[] connections) {
		this.connections = connections;
	}

	/**
	 * get weights (1D array)
	 * @return
	 */
	public final double[] getWeights() {
		final double[] weights = new double[connections.length];

		for (int i = 0; i < connections.length; i++)
			weights[i] = connections[i].getWeight();

		return weights;
	}

	/**
	 * forward pass
	 * @param input
	 * @param weights
	 * @param activationFunction
	 * @return
	 */
	public final double computeFeedforward(final double[] input, final double[] weights, final ActivationFunction activationFunction) {
		double weightedSum = 0;

		for (int i = 0; i < weights.length; i++) {
			weightedSum += input[i] * weights[i];
		}

		//add bias
		weightedSum += bias;

		//apply the activation function
		output = activationFunction.calculateActivation(weightedSum);

		return output;
	}

}
