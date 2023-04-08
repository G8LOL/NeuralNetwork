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

	public Neuron(int numConnections, int numOutput, WeightInitialization weightInitialization) {
		this.output = 0;
		this.error = 0;
		this.bias = Math.random();

		this.connections = new Connection[numOutput];

		for (int i = 0; i < numOutput; i++) {
			this.connections[i] = new Connection(weightInitialization.initializeWeight(numConnections, numOutput));
		}
	}

	public Neuron(int numConnections, int numOutput, WeightInitialization weightInitialization, boolean inputNeuron) {
		this(numConnections, numOutput, weightInitialization);

		this.inputNeuron = inputNeuron;
	}

	/**
	 * get input value
	 * @return
	 */
	public double getInput() {
		return input;
	}

	/**
	 * set input value
	 * @param input
	 */
	public void setInput(double input) {
		this.input = input;
	}

	/**
	 * get output value
	 * @return
	 */
	public double getOutput() {
		return inputNeuron ? input : output;
	}

	/**
	 * set output value
	 * @param output
	 */
	public void setOutput(double output) {
		this.output = output;
	}

	/**
	 * get error value
	 * @return
	 */
	public double getError() {
		return error;
	}

	/**
	 * set error value
	 * @param error
	 */
	public void setError(double error) {
		this.error = error;
	}

	/**
	 * get bias value
	 * @return
	 */
	public double getBias() {
		return bias;
	}

	/**
	 * set bias value
	 * @param bias
	 */
	public void setBias(double bias) {
		this.bias = bias;
	}

	/**
	 * get connections
	 * @return
	 */
	public Connection[] getConnections() {
		return connections;
	}

	/**
	 * set connections
	 * @param connections
	 */
	public void setConnections(Connection[] connections) {
		this.connections = connections;
	}

	/**
	 * get weights (1D array)
	 * @return
	 */
	public double[] getWeights() {
		double[] weights = new double[connections.length];

		for (int i = 0; i < connections.length; i++) {
			weights[i] = connections[i].getWeight();
		}

		return weights;
	}

	/**
	 * forward pass
	 * @param input
	 * @param weights
	 * @param activationFunction
	 * @return
	 */
	public double computeFeedforward(double[] input, double[] weights, ActivationFunction activationFunction) {
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
