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

    public Connection(double weight) {
        this.weight = weight;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

}
