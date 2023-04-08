package dev.g8.neuralnet.functions.activation;

/**
 * @author G8LOL
 * @since 4/7/2023
 */
public enum ActivationFunction {

    LOGISTIC_SIGMOID {
        @Override
        public final double calculateActivation(final double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public final double calculateDerivative(final double x) {
            return (1 - calculateActivation(x)) * calculateActivation(x);
        }
    },
    TANH {
        @Override
        public final double calculateActivation(final double x) {
            return Math.tanh(x);
        }

        @Override
        public final double calculateDerivative(final double x) {
            return 1 - Math.pow(Math.tanh(x), 2);
        }
    },
    RELU {
        @Override
        public final double calculateActivation(final double x) {
            return Math.max(0, x);
        }

        @Override
        public final double calculateDerivative(final double x) {
            return x > 0 ? 1 : 0;
        }
    },
    LEAKY_RELU {
        @Override
        public final double calculateActivation(final double x) {
            return x > 0 ? x : 0.01 * x;
        }

        @Override
        public final double calculateDerivative(final double x) {
            return x > 0 ? 1 : 0.01;
        }
    },
    SOFTPLUS {
        @Override
        public final double calculateActivation(final double x) {
            return Math.log(1 + Math.exp(x));
        }

        @Override
        public final double calculateDerivative(final double x) {
            return 1 / (1 + Math.exp(-x));
        }
    },
    SOFTSIGN {
        @Override
        public final double calculateActivation(final double x) {
            return x / (1 + Math.abs(x));
        }

        @Override
        public final double calculateDerivative(final double x) {
            return 1 / Math.pow(1 + Math.abs(x), 2);
        }
    },
    SINUSOID {
        @Override
        public final double calculateActivation(final double x) {
            return Math.sin(x);
        }

        @Override
        public final double calculateDerivative(final double x) {
            return Math.cos(x);
        }
    },
    SINC {
        @Override
        public final double calculateActivation(final double x) {
            return x == 0 ? 1 : Math.sin(x) / x;
        }

        @Override
        public final double calculateDerivative(final double x) {
            return x == 0 ? 0 : (Math.cos(x) / x) - (Math.sin(x) / Math.pow(x, 2));
        }
    },
    GAUSSIAN {
        @Override
        public final double calculateActivation(final double x) {
            return Math.exp(-Math.pow(x, 2));
        }

        @Override
        public final double calculateDerivative(final double x) {
            return -2 * x * Math.exp(-Math.pow(x, 2));
        }
    },
    BENT_IDENTITY {
        @Override
        public final double calculateActivation(final double x) {
            return (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x;
        }

        @Override
        public final double calculateDerivative(final double x) {
            return x / (2 * Math.sqrt(Math.pow(x, 2) + 1)) + 1;
        }
    },
    BIPOLAR {
        @Override
        public final double calculateActivation(final double x) {
            return x >= 0 ? 1 : -1;
        }

        @Override
        public final double calculateDerivative(final double x) {
            return 0;
        }
    },
    BIPOLAR_SIGMOID {
        @Override
        public final double calculateActivation(final double x) {
            return (2 / (1 + Math.exp(-x))) - 1;
        }

        @Override
        public final double calculateDerivative(final double x) {
            return (1 - Math.pow(calculateActivation(x), 2)) / 2;
        }
    },
    HARD_TANH {
        @Override
        public final double calculateActivation(final double x) {
            return x < -1 ? -1 : x > 1 ? 1 : x;
        }

        @Override
        public final double calculateDerivative(final double x) {
            return x < -1 || x > 1 ? 0 : 1;
        }
    },
    ABSOLUTE {
        @Override
        public final double calculateActivation(final double x) {
            return Math.abs(x);
        }

        @Override
        public final double calculateDerivative(final double x) {
            return x < 0 ? -1 : 1;
        }
    },
    SELU {
        private final double ALPHA = 1.6732632423543772848170429916717;

        @Override
        public final double calculateActivation(final double x) {
            return x > 0 ? x : ALPHA * (Math.exp(x) - 1);
        }

        @Override
        public final double calculateDerivative(final double x) {
            return x > 0 ? 1 : ALPHA * Math.exp(x);
        }
    };

    public abstract double calculateActivation(final double x);

    public abstract double calculateDerivative(final double x);

}
