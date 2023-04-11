package dev.g8.neuralnet.layers.impl;

import dev.g8.neuralnet.components.Neuron;
import dev.g8.neuralnet.functions.activation.ActivationFunction;
import dev.g8.neuralnet.functions.loss.LossFunction;
import dev.g8.neuralnet.initializations.WeightInitialization;
import dev.g8.neuralnet.layers.api.AbstractLayer;

/**
 * @author G8LOL
 * @since 4/5/2023
 */
public final class OutputLayer extends AbstractLayer {

    private final ActivationFunction activationFunction;
    private final LossFunction lossFunction;

    private final Neuron[] neurons;

    private double[] output, outputErrors;

    private AbstractLayer prevLayer, nextLayer;

    private final int numInput, numOutput;

    public OutputLayer(final int numInput, final int numOutput, final ActivationFunction activationFunction, final LossFunction lossFunction, final WeightInitialization weightInitialization) {
        this.numInput = numInput;
        this.numOutput = numOutput;

        this.activationFunction = activationFunction;
        this.lossFunction = lossFunction;

        this.neurons = new Neuron[numInput];

        for (int i = 0; i < numInput; i++) {
            final Neuron neuron = new Neuron(numInput, numOutput, weightInitialization, false);

            neurons[i] = neuron;
        }
    }

    /**
     * get the activation function used by this layer
     * @return
     */
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    @Override
    public final void computeForward(final double[] prevInput, final double[][] weights) {
        //first find the weighted sum
        //weighted_sum = (input_1 * weight_1) + (input_2 * weight_2) + ... + (input_n * weight_n)

        //weights shape = input layer input neurons x hidden layer neuron count
        //eg 2 x 4

        final double[] outputs = new double[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].computeFeedforward(prevInput, weights[i], activationFunction);
        }

        this.output = outputs;
    }

    /**
     * compute the backpropagation for the output layer
     * @param predicted
     * @param desired
     * @return
     */
    public final double[] computeBackprop(final double[] predicted, final double[] desired) {
        /*
         * this calculates the delta/error for each output neuron
         *
         * the calculateDerivative() method uses the chain rule to compute
         * the derivative of the loss function with respect to the predicted output.
         */

        final double[] outputErrors = new double[neurons.length];

        final double[] errors = lossFunction.calculateDerivative(desired, predicted);

        for (int i = 0; i < neurons.length; i++) {
            final double error = errors[i];

            outputErrors[i] = error * activationFunction.calculateDerivative(predicted[i]);
        }

        this.outputErrors = outputErrors;

        return outputErrors;
    }

    /**
     * update the weights of the output layer
     * @param abstractLayer - previous layer
     * @param outputErrors
     * @param hiddenLayerOut
     * @param learningRate
     */
    public final void updateWeights(final AbstractLayer abstractLayer, final double[] outputErrors, final double[] hiddenLayerOut, final double learningRate) {
        final double[][] outputWeights = prevLayer.getWeights();

        //shape is numOutput x numInput because we are going backwards
        for (int i = 0; i < abstractLayer.getNeurons().length; i++) {
            for (int j = 0; j < neurons.length; j++) {
                outputWeights[j][i] += learningRate * outputErrors[j] * hiddenLayerOut[i];
            }
        }

        prevLayer.setWeights(outputWeights);
    }

    /**
     * update the biases of the output layer
     * @param outputErrors
     * @param learningRate
     */
    public final void updateBiases(final double[] outputErrors, final double learningRate) {
        final double[] outputBias = getBias();

        for (int i = 0; i < neurons.length; i++) {
            outputBias[i] += learningRate * outputErrors[i];
        }

        setBias(outputBias);
    }

    @Override
    public final double[] getOutput() {
        return this.output;
    }

    @Override
    public final double[] getOutputErrors() {
        return this.outputErrors;
    }

    @Override
    public final double[] getBias() {
        final double[] bias = new double[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            bias[i] = neurons[i].getBias();
        }

        return bias;
    }

    @Override
    public final Neuron[] getNeurons() {
        return neurons;
    }

    @Override
    public final void setPrevLayer(final AbstractLayer prevLayer) {
        this.prevLayer = prevLayer;
    }

    @Override
    public final AbstractLayer getPrevLayer() {
        return prevLayer;
    }

    @Override
    public final void setNextLayer(final AbstractLayer nextLayer) {
        this.nextLayer = nextLayer;
    }

    @Override
    public final AbstractLayer getNextLayer() {
        return nextLayer;
    }

    @Override
    public final double[][] getWeights() {
        final double[][] weights = new double[numOutput][numInput];

        for (int i = 0; i < numOutput; i++) {
            //columns
            for (int a = 0; a < numInput; a++) {
                //System.out.println(i + " " + a);
                weights[i][a] = neurons[a].getWeights()[i];
            }
        }

        return weights;
    }

    @Override
    public final void setWeights(final double[][] weights) {
        for (int i = 0; i < neurons.length; i++) {
            //columns
            for (int a = 0; a < neurons[i].getConnections().length; a++) {
                //System.out.println(i + " " + a);
                neurons[i].getConnections()[a].setWeight(weights[a][i]);

                //weights[i][a] = neurons[a].getWeights()[i];
            }
        }
    }

    @Override
    public final void setBias(final double[] outputLayerBias) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].setBias(outputLayerBias[i]);
        }
    }
}