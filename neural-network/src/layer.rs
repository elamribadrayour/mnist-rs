use crate::neuron::Neuron;
use rand::RngCore;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(rng: &mut dyn RngCore, output_size: usize, input_size: usize) -> Self {
        Self {
            neurons: (0..output_size)
                .map(|_| Neuron::new(rng, input_size))
                .collect(),
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.neurons.iter().map(|n| n.forward(input)).collect()
    }

    pub fn backprop(&mut self, input: &[f32], error: f32, learning_rate: f32) {
        self.neurons
            .iter_mut()
            .for_each(|n| n.backprop(input, error, learning_rate));
    }
}
