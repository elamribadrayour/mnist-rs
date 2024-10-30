use anyhow::Result;
use neural_network::MLP;
use rand::thread_rng;

fn main() -> Result<()> {
    let mut network = MLP::new(&mut thread_rng(), vec![784, 100, 60, 10]);
    let inputs = vec![vec![0.0; 784]; 1_000];
    let outputs = vec![vec![0.0; 10]; 1_000];

    network.train(&inputs, &outputs, 10, 0.1);

    Ok(())
}
