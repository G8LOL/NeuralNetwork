package dev.g8.neuralnet.network.api;

import dev.g8.neuralnet.layers.api.AbstractLayer;
import dev.g8.neuralnet.optimizations.OptimizationAlgorithm;
import dev.g8.neuralnet.utils.objects.DataSet;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

/**
 * @author G8LOL
 * @since 4/7/2023
 */
public abstract class AbstractNetwork {

    protected List<AbstractLayer> layers;
    protected double learningRate;
    protected OptimizationAlgorithm optimizationAlgorithm;

    /**
     * setup network (e.g setup layers)
     */
    public abstract void setup();

    /**
     * train/fit network
     * @param dataSet
     * @param epochs
     * @param batchSize
     */
    public abstract void train(final DataSet dataSet, final int epochs, final int batchSize);

    /**
     * iterate through one set of data
     * @param input
     * @param output
     */
    protected abstract void iterate(final double[] input, final double[] output);

    /**
     * predict output from input
     * @param input
     * @return
     */
    public abstract DataSet predict(final DataSet input);

    /**
     * print network info
     */
    public final void printNetwork() {
        System.out.println("Network: " + getClass().getSimpleName() + " - " + new Date());
        System.out.println("Num Layers: " + layers.size());
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Optimization algorithm: " + optimizationAlgorithm);

        System.out.println("Layers: ");
        for (final AbstractLayer layer : layers)
            System.out.println("\t" + layer.getClass().getSimpleName());
    }

    public final static class NetworkBuilder {
        private final List<AbstractLayer> layers = new ArrayList<>();
        private double learningRate;
        private OptimizationAlgorithm optimizationAlgorithm;

        private final Class<? extends AbstractNetwork> clazz;

        public NetworkBuilder(final Class<? extends AbstractNetwork> clazz) {
            this.clazz = clazz;
        }

        public final NetworkBuilder withLayers(final AbstractLayer... layers) {
            this.layers.addAll(Arrays.asList(layers));

            return this;
        }

        public final NetworkBuilder withLayer(final AbstractLayer layer) {
            this.layers.add(layer);

            return this;
        }

        public final NetworkBuilder withOptimizationAlgorithm(final OptimizationAlgorithm optimizationAlgorithm) {
            this.optimizationAlgorithm = optimizationAlgorithm;

            return this;
        }

        public final NetworkBuilder withLearningRate(final double learningRate) {
            this.learningRate = learningRate;

            return this;
        }

        public final AbstractNetwork build() {
            try {
                final Constructor<? extends AbstractNetwork> constructor = clazz.getDeclaredConstructor();
                constructor.setAccessible(true);

                final AbstractNetwork network = constructor.newInstance();
                network.layers = layers;
                network.learningRate = learningRate;
                network.optimizationAlgorithm = optimizationAlgorithm;

                return network;
            } catch (Exception e) {
                e.printStackTrace();
            }

            return null;
        }
    }

}
