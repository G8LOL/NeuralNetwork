package dev.g8.neuralnet.functions.loss;

/**
 * @author G8LOL
 * @since 4/7/2023
 */
public enum LossFunction {

    MEAN_SQUARED_ERROR {
        @Override
        public final double calculateLoss(final double[] predicted, final double[] desired) {
            double sum = 0;

            for (int i = 0; i < predicted.length; i++) {
                sum += Math.pow(predicted[i] - desired[i], 2);
            }

            return sum / predicted.length;
        }

        @Override
        public final double[] calculateDerivative(final double[] predicted, final double[] desired) {
            final double[] derivative = new double[predicted.length];

            for (int i = 0; i < predicted.length; i++) {
                derivative[i] = 2 * (predicted[i] - desired[i]) / predicted.length;
            }

            return derivative;
        }
    },
    CROSS_ENTROPY {
        @Override
        public final double calculateLoss(final double[] predicted, final double[] desired) {
            double sum = 0;

            for (int i = 0; i < predicted.length; i++) {
                sum += desired[i] * Math.log(predicted[i]) + (1 - desired[i]) * Math.log(1 - predicted[i]);
            }

            return -sum / predicted.length;
        }

        @Override
        public final double[] calculateDerivative(final double[] predicted, final double[] desired) {
            final double[] derivative = new double[predicted.length];

            for (int i = 0; i < predicted.length; i++) {
                derivative[i] = (predicted[i] - desired[i]) / (predicted[i] * (1 - predicted[i]));
            }

            return derivative;
        }
    },
    BINARY_CROSS_ENTROPY {
        @Override
        public final double calculateLoss(final double[] predicted, final double[] desired) {
            double sum = 0;

            for (int i = 0; i < predicted.length; i++) {
                sum += desired[i] * Math.log(predicted[i]) + (1 - desired[i]) * Math.log(1 - predicted[i]);
            }

            return -sum;
        }

        @Override
        public final double[] calculateDerivative(final double[] predicted, final double[] desired) {
            final double[] derivative = new double[predicted.length];

            for (int i = 0; i < predicted.length; i++) {
                derivative[i] = (predicted[i] - desired[i]) / (predicted[i] * (1 - predicted[i]));
            }

            return derivative;
        }
    },
    HINGE {
        @Override
        public final double calculateLoss(final double[] predicted, final double[] desired) {
            double sum = 0;

            for (int i = 0; i < predicted.length; i++) {
                sum += Math.max(0, 1 - desired[i] * predicted[i]);
            }

            return sum / predicted.length;
        }

        @Override
        public final double[] calculateDerivative(final double[] predicted, final double[] desired) {
            final double[] derivative = new double[predicted.length];

            for (int i = 0; i < predicted.length; i++) {
                derivative[i] = desired[i] * predicted[i] < 1 ? -desired[i] : 0;
            }

            return derivative;
        }
    },
    SQUARED_HINGE {
        @Override
        public final double calculateLoss(final double[] predicted, final double[] desired) {
            double sum = 0;

            for (int i = 0; i < predicted.length; i++) {
                sum += Math.pow(Math.max(0, 1 - desired[i] * predicted[i]), 2);
            }

            return sum / predicted.length;
        }

        @Override
        public final double[] calculateDerivative(final double[] predicted, final double[] desired) {
            final double[] derivative = new double[predicted.length];

            for (int i = 0; i < predicted.length; i++) {
                derivative[i] = desired[i] * predicted[i] < 1 ? -2 * desired[i] * Math.max(0, 1 - desired[i] * predicted[i]) : 0;
            }

            return derivative;
        }
    },
    KULLBACK_LEIBLER_DIVERGENCE {
        @Override
        public final double calculateLoss(final double[] predicted, final double[] desired) {
            double sum = 0;

            for (int i = 0; i < predicted.length; i++) {
                sum += desired[i] * Math.log(desired[i] / predicted[i]);
            }

            return sum;
        }

        @Override
        public final double[] calculateDerivative(final double[] predicted, final double[] desired) {
            final double[] derivative = new double[predicted.length];

            for (int i = 0; i < predicted.length; i++) {
                derivative[i] = desired[i] / predicted[i];
            }

            return derivative;
        }
    },
    POISSON {
        @Override
        public final double calculateLoss(final double[] predicted, final double[] desired) {
            double sum = 0;

            for (int i = 0; i < predicted.length; i++) {
                sum += desired[i] * Math.log(predicted[i]) - predicted[i];
            }

            return sum;
        }

        @Override
        public final double[] calculateDerivative(final double[] predicted, final double[] desired) {
            final double[] derivative = new double[predicted.length];

            for (int i = 0; i < predicted.length; i++) {
                derivative[i] = desired[i] / predicted[i] - 1;
            }

            return derivative;
        }
    },
    COSINE_PROXIMITY {
        @Override
        public final double calculateLoss(final double[] predicted, final double[] desired) {
            double sum = 0;

            for (int i = 0; i < predicted.length; i++) {
                sum += predicted[i] * desired[i];
            }

            return 1 - sum / (Math.sqrt(sum(predicted)) * Math.sqrt(sum(desired)));
        }

        @Override
        public final double[] calculateDerivative(final double[] predicted, final double[] desired) {
            final double[] derivative = new double[predicted.length];

            for (int i = 0; i < predicted.length; i++) {
                derivative[i] = -desired[i] / (Math.sqrt(sum(predicted)) * Math.sqrt(sum(desired))) + predicted[i] * sum(desired) / (Math.pow(Math.sqrt(sum(predicted)), 3) * Math.sqrt(sum(desired)));
            }

            return derivative;
        }
    };

    public final double sum(final double[] array) {
        double sum = 0;

        for (double v : array) {
            sum += v;
        }

        return sum;
    }

    public abstract double calculateLoss(final double[] predicted, final double[] desired);

    public abstract double[] calculateDerivative(final double[] predicted, final double[] desired);

}
