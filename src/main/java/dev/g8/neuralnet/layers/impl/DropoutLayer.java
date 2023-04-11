package dev.g8.neuralnet.layers.impl;

import dev.g8.neuralnet.components.Neuron;
import dev.g8.neuralnet.layers.api.AbstractLayer;

/**
 * basically a dropout layer is a regularization technique that just sets the output of some neurons to 0 (drops them out)
 * this is done to prevent overfitting (i.e. the model memorizes the training data instead of learning the general pattern)
 * <p>
 * the dropout rate is the percentage of neurons that will be dropped out
 * in the forward pass, the output of dropped neurons -> 0
 * in the backward pass, the error of dropped neurons -> 0
 * <p>
 * the output of the neurons that are not dropped out will be multiplied by the inverted dropout rate (scale up)
 * scaling is done so that the expected value of the output of the layer is the same as the output of the layer without dropout
 * <p>
 * dropout method is inverted which is what most frameworks use
 * <p>
 *
 * @author G8LOL
 * @since 4/9/2023
 */
public class DropoutLayer extends AbstractLayer {

    private final double dropoutRate, invertedRate;

    private final int[] droppedOutNeurons;

    private double[] output, outputErrors;

    private AbstractLayer prevLayer, nextLayer;

    public DropoutLayer(final int numPrevLayerNodes, final double dropoutRate) {
        this.dropoutRate = dropoutRate;

        this.invertedRate = 1.0 / (1.0 - dropoutRate);

        droppedOutNeurons = new int[numPrevLayerNodes];
    }

    @Override
    public final void computeForward(double[] prevInput, double[][] weights) {
        switch (mode) {
            case TRAINING -> {
                //select random neurons to drop out
                for (int i = 0; i < prevInput.length; i++) {
                    if (Math.random() < dropoutRate) {
                        prevInput[i] = 0;

                        droppedOutNeurons[i] = 0;
                    } else {
                        prevInput[i] *= invertedRate;

                        droppedOutNeurons[i] = 1;
                    }
                }

                output = prevInput;
            }
            case PREDICTION -> {
                //set all neurons to be active
                for (int i = 0; i < prevInput.length; i++)
                    droppedOutNeurons[i] = 1;

                output = prevInput;
            }
        }
    }

    public final void computeBackprop(final double[] prevOutputErrors) {
        final double[] outputErrors = new double[prevOutputErrors.length];

        for (int i = 0; i < prevOutputErrors.length; i++) {
            if (droppedOutNeurons[i] == 1)
                outputErrors[i] = prevOutputErrors[i] * invertedRate;
            else
                outputErrors[i] = 0;
        }

        this.outputErrors = outputErrors;
    }

    @Override
    public final double[] getOutput() {
        return this.output;
    }

    @Override
    public final double[] getBias() {
        return prevLayer.getBias();
    }

    @Override
    public final double[] getOutputErrors() {
        return outputErrors;
    }

    @Override
    public final double[][] getWeights() {
        return prevLayer.getWeights();
    }

    @Override
    public final void setWeights(final double[][] weights) {
        //set the weights of the previous layer to the weights provided (required during backpropagation)
        prevLayer.setWeights(weights);
    }

    @Override
    public final void setBias(final double[] hiddenLayerBias) {
        //set the bias of the previous layer to the bias provided (required during backpropagation)
        prevLayer.setBias(hiddenLayerBias);
    }

    @Override
    public final Neuron[] getNeurons() {
        //returns neurons of the next layer because during backprop dense layer requires the neurons of the next layer
        //to compute the errors/gradients and dropout layer doesn't have any neurons
        return nextLayer.getNeurons();
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

    /*
    * remove for now

    public final void updateWeights(AbstractLayer prevLayer, double[] dropoutErrors, double[] output, double learningRate) {
        final double[][] weights = getWeights();

        for (int i = 0; i < weights.length; i++) {
            for (int a = 0; a < weights[i].length; a++) {
                weights[i][a] += learningRate * dropoutErrors[a] * output[i];
            }
        }

        setWeights(weights);
    }

    public final void updateBiases(double[] dropoutErrors, double learningRate) {
        final double[] bias = getBias();

        for (int i = 0; i < bias.length; i++) {
            bias[i] += learningRate * dropoutErrors[i];
        }

        setBias(bias);
    }*/
}
