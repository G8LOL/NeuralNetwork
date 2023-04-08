package dev.g8.neuralnet.layers.api;

import dev.g8.neuralnet.components.Neuron;
import dev.g8.neuralnet.functions.activation.ActivationFunction;
import dev.g8.neuralnet.functions.loss.LossFunction;
import dev.g8.neuralnet.initializations.WeightInitialization;
import dev.g8.neuralnet.layers.impl.OutputLayer;
import dev.g8.neuralnet.network.api.AbstractNetwork;

import java.lang.reflect.Constructor;

/**
 * @author G8LOL
 * @since 4/2/2023
 */
public abstract class AbstractLayer {

    public abstract void computeForward(final double[] input, final double[][] weights);

    public abstract double[][] getWeights();

    public abstract void setWeights(final double[][] outputLayerWeights);

    public abstract void setBias(final double[] outputLayerBias);

    public abstract double[] getBias();

    public abstract double[] getOutput();

    public abstract Neuron[] getNeurons();

    public double[] getOutputErrors() {
        return new double[0];
    }

    public AbstractLayer getPrevLayer() {
        return null;
    }

    public abstract void setPrevLayer(final AbstractLayer abstractLayer);

    public AbstractLayer getNextLayer() {
        return null;
    }

    public abstract void setNextLayer(final AbstractLayer abstractLayer);

}
