package dev.g8.neuralnet.layers.impl;

import dev.g8.neuralnet.components.Neuron;
import dev.g8.neuralnet.functions.activation.ActivationFunction;
import dev.g8.neuralnet.initializations.WeightInitialization;
import dev.g8.neuralnet.layers.api.AbstractLayer;

/**
 * @author G8LOL
 * @since 4/5/2023
 */
public final class DenseHiddenLayer extends AbstractLayer {

    private final ActivationFunction activationFunction;

    private final Neuron[] neurons;

    private double[] output, outputErrors;

    private final int numInput, numOutput;

    private AbstractLayer prevLayer, nextLayer;

    public DenseHiddenLayer(final int numInput, final int numOutput, final ActivationFunction activationFunction, final WeightInitialization weightInitialization) {
        this.activationFunction = activationFunction;

        this.numInput = numInput;
        this.numOutput = numOutput;

        neurons = new Neuron[numInput];

        for (int i = 0; i < numInput; i++) {
            final Neuron neuron = new Neuron(numInput, numOutput, weightInitialization,false);

            neurons[i] = neuron;
        }
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

    /**
     * get the activation function for this layer
     * @return
     */
    public final ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    @Override
    public final void computeForward(final double[] prevInput, final double[][] weights) {
        //first find the weighted sum
        //weighted_sum = (input_1 * weight_1) + (input_2 * weight_2) + ... + (input_n * weight_n)

        //weights shape = input layer input neurons x hidden layer neuron count
        //eg 2 x 4
        double[] outputs = new double[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].computeFeedforward(prevInput, weights[i], activationFunction);
        }

        this.output = outputs;
    }

    /**
     * Compute the output errors for the hidden layer
     * <p>
     * note: nextLayer used to be outputLayer
     * @param nextLayer
     * @param hiddenLayerOut
     * @param outputErrors
     * @return
     */
    public final double[] computeBackprop(final AbstractLayer nextLayer, final double[] hiddenLayerOut, final double[] outputErrors) {
        /*
         * same thing as output layer but calculate error using the output layers weights
         *
         * error variable is the sum of the output delta times the corresponding weight
         *
         * im using normal gradient descent for now ＞﹏＜
         */
        final double[] hiddenErrors = new double[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            double error = 0.0;
            for (int j = 0; j < nextLayer.getNeurons().length; j++) {
                error += outputErrors[j] * nextLayer.getPrevLayer().getWeights()[j][i];
            }
            hiddenErrors[i] = error * activationFunction.calculateDerivative(hiddenLayerOut[i]);
        }

        this.outputErrors = hiddenErrors;

        return hiddenErrors;
    }

    /**
     * Update the weights for the hidden layer
     * @param prevLayer
     * @param hiddenErrors
     * @param inputLayerOut
     * @param learningRate
     */
    public final void updateWeights(final AbstractLayer prevLayer, final double[] hiddenErrors, final double[] inputLayerOut, final double learningRate) {
        final double[][] hiddenWeights = prevLayer.getWeights();

        for (int i = 0; i < prevLayer.getNeurons().length; i++) {
            for (int j = 0; j < neurons.length; j++) {
                hiddenWeights[j][i] += learningRate * hiddenErrors[j] * inputLayerOut[i];
            }
        }

        prevLayer.setWeights(hiddenWeights);
    }

    /**
     * Update the biases for the hidden layer
     * @param hiddenErrors
     * @param learningRate
     */
    public final void updateBiases(final double[] hiddenErrors, final double learningRate) {
        final double[] hiddenBias = getBias();

        for (int i = 0; i < neurons.length; i++) {
            hiddenBias[i] += learningRate * hiddenErrors[i];
        }

        setBias(hiddenBias);
    }

    @Override
    public final double[] getOutput() {
        return this.output;
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
    public final double[] getOutputErrors() {
        return outputErrors;
    }

    @Override
    public final double[][] getWeights() {
        // rows    columns
        // numOutput is the number of neurons in the next layer :3
        // numInput is the number of neurons in the current layer
        //this will allow us to create a matrix with the correct dimensions of all the weights
        // eg 4 neurons in current layer, 2 neurons in next layer
        // 2 x 4 matrix
        // 8 weights total
        // example table:

        // 0 1 2 3
        // 4 5 6 7

        final double[][] weights = new double[numOutput][numInput];

        //so each input neuron will have numOutput amount of connections connecting to neurons in the hidden layer

        //System.out.println("aa " + numInput + " " + numOutput);

        //rows
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
            }
        }
    }

    @Override
    public final void setBias(final double[] hiddenLayerBias) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].setBias(hiddenLayerBias[i]);
        }
    }
}
