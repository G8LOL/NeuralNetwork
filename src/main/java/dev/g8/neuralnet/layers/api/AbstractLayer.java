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

    protected Mode mode;

    /**
     * forward propagation
     * @param input
     * @param weights
     */
    public abstract void computeForward(final double[] input, final double[][] weights);

    /**
     * get weights in shape of [numOut][numIn]
     * @return
     */
    public abstract double[][] getWeights();

    /**
     * set weights in shape of [numOut][numIn]
     * @param outputLayerWeights
     */
    public abstract void setWeights(final double[][] outputLayerWeights);

    /**
     * set bias in shape of [numOut]
     * @param outputLayerBias
     */
    public abstract void setBias(final double[] outputLayerBias);

    /**
     * get bias in shape of [numOut]
     * @return
     */
    public abstract double[] getBias();

    /**
     * get output
     * @return
     */
    public abstract double[] getOutput();

    /**
     * get neurons
     * @return
     */
    public abstract Neuron[] getNeurons();

    /**
     * get out errors
     * @return
     */
    public double[] getOutputErrors() {
        return new double[0];
    }

    /**
     * get previus layer
     * @return
     */
    public AbstractLayer getPrevLayer() {
        return null;
    }

    /**
     * set previous layer
     * @param abstractLayer
     */
    public abstract void setPrevLayer(final AbstractLayer abstractLayer);

    /**
     * get next layer
     * @return
     */
    public AbstractLayer getNextLayer() {
        return null;
    }

    /**
     * set next layer
     * @param abstractLayer
     */
    public abstract void setNextLayer(final AbstractLayer abstractLayer);

    /**
     * get mode
     * @return
     */
    public Mode getMode() {
        return mode;
    }

    /**
     * set mode
     * @param mode
     */
    public void setMode(final Mode mode) {
        this.mode = mode;
    }

}
