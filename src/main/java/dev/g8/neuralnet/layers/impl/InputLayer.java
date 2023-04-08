package dev.g8.neuralnet.layers.impl;

import dev.g8.neuralnet.components.Neuron;
import dev.g8.neuralnet.initializations.WeightInitialization;
import dev.g8.neuralnet.layers.api.AbstractLayer;

/**
 * @author G8LOL
 * @since 4/5/2023
 */
public final class InputLayer extends AbstractLayer {

    private final Neuron[] neurons;

    private final int numInput, numOutput;

    private AbstractLayer prevLayer, nextLayer;

    public InputLayer(int numInput, int numOutput, WeightInitialization weightInitialization) {
        //input number would be the amount of features (e.g 2 in XOR example)

        //output is basically the amount of input neurons for the hidden layer (e.g 4)
        //which we need for the amount of connections/synapses we need to create

        this.numInput = numInput;
        this.numOutput = numOutput;

        this.neurons = new Neuron[numInput];

        for (int i = 0; i < numInput; i++) {
            Neuron neuron = new Neuron(numInput, numOutput, weightInitialization, true);

            neurons[i] = neuron;
        }
    }

    @Override
    public final void computeForward(double[] input, double[][] weights) {
        int index = 0;

        for (Neuron neuron : neurons) {
            neuron.setInput(input[index]);

            index++;
        }
    }

    @Override
    public final double[] getOutput() {
        double[] output = new double[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            output[i] = neurons[i].getOutput();
        }

        return output;
    }

    @Override
    public final double[][] getWeights() {
        //double[] outputWeights = new double[neurons.w]
        double[][] weights = new double[numOutput][numInput];

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
    public final void setWeights(double[][] weights) {
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
    public final void setBias(double[] outputLayerBias) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].setBias(outputLayerBias[i]);
        }
    }

    @Override
    public final double[] getBias() {
        double[] bias = new double[neurons.length];

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
    public final void setPrevLayer(AbstractLayer prevLayer) {
        this.prevLayer = prevLayer;
    }

    @Override
    public final AbstractLayer getPrevLayer() {
        return prevLayer;
    }

    @Override
    public final void setNextLayer(AbstractLayer nextLayer) {
        this.nextLayer = nextLayer;
    }

    @Override
    public final AbstractLayer getNextLayer() {
        return nextLayer;
    }
}
