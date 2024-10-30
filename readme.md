# MNIST Neural Network

Welcome to the MNIST Neural Network Project! This project implements a simple neural network library in Rust, including layers, neurons, and activation functions. The library is designed to be easy to use and extend, making it a great starting point for anyone interested in learning about neural networks and machine learning.

## Features

- **Layer and Neuron Implementation**: The core components of a neural network, including layers and neurons, are implemented with forward and backpropagation methods.
- **Activation Functions**: Includes a Sigmoid activation function, with the ability to add more activation functions as needed.
- **Training**: Supports training of the neural network with adjustable learning rates and epochs.
- **MNIST Dataset**: Train the model using the MNIST dataset, a large database of handwritten digits commonly used for training various image processing systems.

## File Structure

- `src/`
  - `activations.rs`: Contains the activation functions used in the neural network.
  - `layer.rs`: Defines the `Layer` struct and its associated methods.
  - `mlp.rs`: Implements the multi-layer perceptron (MLP) structure.
  - `neuron.rs`: Defines the `Neuron` struct and its associated methods.
  - `lib.rs`: The main library file that ties all the components together.

- `main/`
  - `main.rs`: Example usage of the neural network library, including training a network with sample data.
  - `Cargo.toml`: Configuration file for the main project, including dependencies.

## Getting Started

### Prerequisites

- Rust (https://www.rust-lang.org/tools/install)
- Cargo (comes with Rust)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/neural-network.git
   cd neural-network
   ```

2. Build the project:
   ```sh
   cargo build
   ```

3. Run the example:
   ```sh
   cargo run --bin main
   ```

## Usage

To use the neural network library in your own project, add it as a dependency in your `Cargo.toml`:


