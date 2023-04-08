package dev.g8.neuralnet.utils.objects;

import java.util.ArrayList;
import java.util.List;

/**
 * data set object
 *
 * @author G8LOL
 * @since 3/31/2023
 */
public final class DataSet {

    private List<double[]> inputs = new ArrayList<>();
    private List<double[]> outputs = new ArrayList<>();

    public DataSet() {
    }

    public DataSet(final double[][] input, final double[][] output) {
        for (int i = 0; i < input.length; i++) {
            inputs.add(input[i]);
            outputs.add(output[i]);
        }
    }

    /**
     * Adds an input and output
     * @param input
     * @param output
     */
    public final void add(final double[] input, final double[] output) {
        inputs.add(input);
        outputs.add(output);
    }

    /**
     * Adds a 2D array of inputs and outputs
     * @param input
     * @param output
     */
    public final void add(final double[][] input, final double[][] output) {
        for (int i = 0; i < input.length; i++) {
            inputs.add(input[i]);
            outputs.add(output[i]);
        }
    }

    /**
     * Returns the number of inputs
     * @param index
     * @return input
     */
    public final double[] getInput(final int index) {
        return inputs.get(index);
    }

    /**
     * Returns the output at the specified index
     * @param index
     * @return output
     */
    public final double[] getOutput(final int index) {
        return outputs.get(index);
    }

    /**
     * Returns the inputs as a DataSet
     * @return dataset
     */
    public final DataSet getInputs() {
        final DataSet inputs = new DataSet();

        for (final double[] input : this.inputs) {
            inputs.add(input, new double[0]);
        }

        return inputs;
    }

    /**
     * Returns the inputs as a 2D array
     * @return array
     */
    public final double[][] getInputsArray() {
        final double[][] inputs = new double[this.inputs.size()][this.inputs.get(0).length];

        for (int i = 0; i < this.inputs.size(); i++) {
            inputs[i] = this.inputs.get(i);
        }

        return inputs;
    }

    /**
     * Returns the outputs as a DataSet
     * @return dataset
     */
    public final DataSet getOutputs() {
        final DataSet outputs = new DataSet();

        for (final double[] output : this.outputs) {
            outputs.add(new double[0], output);
        }

        return outputs;
    }

    /**
     * Returns the outputs as a 2D array
     * @return array
     */
    public final double[][] getOutputsArray() {
        final double[][] outputs = new double[this.outputs.size()][this.outputs.get(0).length];

        for (int i = 0; i < this.outputs.size(); i++) {
            outputs[i] = this.outputs.get(i);
        }

        return outputs;
    }

    /**
     * Randomly shuffles the dataset
     *
     * used for optimization algorithms
     * @see dev.g8.neuralnet.optimizations.OptimizationAlgorithm
     */
    public final void shuffle() {
        final List<double[]> shuffledInputs = new ArrayList<>();
        final List<double[]> shuffledOutputs = new ArrayList<>();

        while (inputs.size() > 0) {
            final int index = (int) (Math.random() * inputs.size());
            shuffledInputs.add(inputs.remove(index));
            shuffledOutputs.add(outputs.remove(index));
        }

        inputs = shuffledInputs;
        outputs = shuffledOutputs;
    }

    /**
     * Returns the size of the dataset
     * @return size
     */
    public final int size() {
        return inputs.size();
    }

    /**
     * Returns a sub list of the dataset
     * @param start
     * @param end
     * @return dataset
     */
    public final DataSet subList(final int start, final int end) {
        final DataSet subList = new DataSet();

        for (int i = start; i < end; i++)
            subList.add(inputs.get(i), outputs.get(i));

        return subList;
    }

    /**
     * Splits the dataset into datasets based on the ratios
     * e.g. {0.5, 0.5} will split the dataset into two datasets
     * @param ratios
     * @return datasets
     */
    public final DataSet[] split(final double[] ratios) {
        final DataSet[] sets = new DataSet[ratios.length];

        for (int i = 0; i < ratios.length; i++)
            sets[i] = new DataSet();

        int index = 0;

        for (int i = 0; i < inputs.size(); i++) {
            sets[index].add(inputs.get(i), outputs.get(i));

            index++;
            if (index >= sets.length)
                index = 0;
        }

        return sets;
    }

    /**
     * Prints the dataset
     */
    public final void print() {
        for (int i = 0; i < inputs.size(); i++) {
            System.out.print("Input: ");
            for (int j = 0; j < inputs.get(i).length; j++)
                System.out.print(inputs.get(i)[j] + " ");

            System.out.print("Output: ");
            for (int j = 0; j < outputs.get(i).length; j++)
                System.out.print(outputs.get(i)[j] + " ");

            System.out.println();
        }
    }

}
