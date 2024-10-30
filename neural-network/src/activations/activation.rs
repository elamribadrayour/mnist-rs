pub trait Activation {
    fn activate(&self, x: f32) -> f32;
    fn derivative(&self, x: f32) -> f32;
}
