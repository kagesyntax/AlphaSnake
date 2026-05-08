# AlphaSnake

AlphaSnake is a highly performant, AI-driven Snake simulation and training environment built with Rust and Dioxus. It features a persistent neural "brain," real-time swarm-based population evolution, and a comprehensive benchmarking suite.

## 🚀 Features

- **AI-Powered Gameplay:** Controlled by a neural "PolicyBrain" that learns and adapts over time.
- **Swarm Training:** Implements a population-based training system ("Swarm") to evolve smarter snake behaviors through parallel simulations.
- **Persistent State:** Saves game statistics and neural network weights to binary files (`snake_brain.bin`, `snake_stats.bin`) for continuous learning.
- **Interactive Lab:** A built-in dashboard for monitoring population fitness, replaying game traces, and managing model checkpoints.
- **Benchmarking CLI:** Compare different versions of the AI brain directly from the terminal.
- **Cross-Platform:** Runs seamlessly on Web (WASM) and Desktop (native).
- **Modern UI:** Styled with Tailwind CSS for a sleek, responsive interface.

## 🏗️ Architecture

- **`src/ai.rs`**: Neural network logic and move prediction.
- **`src/game.rs`**: Core Snake game engine and rules.
- **`src/swarm.rs`**: Population management and parallel training orchestration.
- **`src/persistence.rs`**: Logic for saving/loading models and checkpoints.
- **`src/evals.rs`**: Tools for evaluating and comparing AI performance.
- **`src/main.rs`**: Dioxus application entry point and UI routing.

## 🛠️ Getting Started

### Prerequisites

- **Rust:** Install via [rustup.rs](https://rustup.rs).
- **Dioxus CLI:** `cargo install dioxus-cli`.

### Running the App

Start the simulation in your browser:
```bash
dx serve --platform web
```

Start the native desktop version:
```bash
dx serve --platform desktop
```

### Benchmarking

Compare the current brain against a specific checkpoint:
```bash
cargo run -- benchmark current artifacts/checkpoints/current
```

## 🧪 Documentation

Detailed technical guides can be found in the `docs/` directory:
- [Checkpoint Manager](docs/checkpoint-manager.md)
- [Model Registry](docs/model-registry.md)
- [Curriculum Arenas](docs/curriculum-arenas.md)
- [Replay and Evaluations](docs/replay-and-evals.md)

## 📜 License

This project is licensed under the MIT License.
