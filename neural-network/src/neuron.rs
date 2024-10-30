use crate::activations::Activation;
use crate::activations::Sigmoid;
use rand::{Rng, RngCore};

pub struct Neuron {
    weights: Vec<f32>,
    bias: f32,
    activation: Box<dyn Activation>,
}

impl Neuron {
    pub fn new(rng: &mut dyn RngCore, input_size: usize) -> Self {
        Self {
            weights: (0..input_size).map(|_| rng.gen_range(0.0..1.0)).collect(),
            bias: rng.gen_range(0.0..1.0),
            activation: Box::new(Sigmoid),
        }
    }

    pub fn forward(&self, input: &[f32]) -> f32 {
        let sum = self
            .weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>()
            + self.bias;
        self.activation.activate(sum)
    }

    // Update weights and bias based on error
    pub fn backprop(&mut self, input: &[f32], error: f32, learning_rate: f32) {
        let sum = self
            .weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>()
            + self.bias;
        let derivative = self.activation.derivative(sum);
        let delta = error * derivative;

        self.weights = self
            .weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w + learning_rate * delta * x)
            .collect();
        self.bias += learning_rate * delta;
    }
}
