use crate::layer::Layer;
use rand::RngCore;

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(rng: &mut dyn RngCore, layers: Vec<usize>) -> Self {
        Self {
            layers: layers
                .iter()
                .zip(layers.iter().skip(1))
                .map(|(prev, next)| Layer::new(rng, *next, *prev))
                .collect(),
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = input.to_vec();
        self.layers
            .iter()
            .for_each(|layer| output = layer.forward(&output));
        output
    }

    pub fn backprop(&mut self, input: &[f32], error: f32, learning_rate: f32) {
        self.layers.iter_mut().rev().for_each(|layer| {
            layer.backprop(input, error, learning_rate);
        });
    }

    pub fn train(
        &mut self,
        inputs: &[Vec<f32>],
        outputs: &[Vec<f32>],
        epochs: usize,
        learning_rate: f32,
    ) {
        for epoch in 0..epochs {
            let mut mean_error = 0.0;
            for (input, output) in inputs.iter().zip(outputs.iter()) {
                let prediction = self.forward(input);
                let error = output
                    .iter()
                    .zip(prediction.iter())
                    .map(|(o, p)| o - p)
                    .collect::<Vec<f32>>();
                mean_error = error.iter().sum::<f32>() / error.len() as f32;
                self.backprop(input, mean_error, learning_rate);
            }
            println!("epoch: {}, mean_error: {}", epoch, mean_error);
        }
    }
}
