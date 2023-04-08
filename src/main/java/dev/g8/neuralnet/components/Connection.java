package dev.g8.neuralnet.components;

import java.util.Random;

/**
 * @author G8LOL
 * @since 3/31/2023
 */
public final class Connection {

    private double weight;

    public Connection() {
        this.weight = (new Random()).nextDouble() - 0.5;
    }

    public Connection(final double weight) {
        this.weight = weight;
    }

    public final double getWeight() {
        return weight;
    }

    public final void setWeight(final double weight) {
        this.weight = weight;
    }

}
