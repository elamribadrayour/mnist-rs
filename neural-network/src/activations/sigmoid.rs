use crate::Activation;

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f32) -> f32 {
        let s = self.activate(x);
        s * (1.0 - s)
    }
}
