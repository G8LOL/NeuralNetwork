package dev.g8.neuralnet.initializations;

import java.util.Random;

/**
 * note: std_dev is the standard deviation
 *
 * @author G8LOL
 * @since 4/8/2023
 */
public enum WeightInitialization {

    RANDOM {
        @Override
        public final double initializeWeight(final int numInput, final int numOutput) {
            return RAND.nextDouble() - 0.5;
        }
    },
    XAVIER {
        @Override
        public final double initializeWeight(final int numInput, final int numOutput) {
            final double variance = 2.0 / (numInput + numOutput);
            final double std_dev = Math.sqrt(variance);

            return RAND.nextGaussian() * std_dev;
        }
    },
    HE {
        @Override
        public final double initializeWeight(final int numInput, final int numOutput) {
            final double variance = 2.0 / numInput;
            final double std_dev = Math.sqrt(variance);

            return RAND.nextGaussian() * std_dev;
        }
    },
    LECUN {
        @Override
        public final double initializeWeight(final int numInput, final int numOutput) {
            final double variance = 1.0 / numInput;
            final double std_dev = Math.sqrt(variance);

            return RAND.nextGaussian() * std_dev;
        }
    },
    UNIFORM {
        @Override
        public final double initializeWeight(final int numInput, final int numOutput) {
            final double lower_bound = -1.0 / Math.sqrt(numInput);
            final double upper_bound = 1.0 / Math.sqrt(numInput);

            return RAND.nextDouble() * (upper_bound - lower_bound) + lower_bound;
        }
    },
    IDENTITY {
        @Override
        public final double initializeWeight(final int numInput, final int numOutput) {
            return 1.0;
        }
    },
    NORMAL {
        @Override
        public final double initializeWeight(final int numInput, final int numOutput) {
            return RAND.nextGaussian();
        }
    };

    private final static Random RAND = new Random();

    public abstract double initializeWeight(final int numInput, final int numOutput);
}
