use crate::game::{Direction, GameState, Pos};
use rand::rngs::SmallRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const OBSERVATION_SIZE: usize = 34;
pub const HIDDEN_SIZE: usize = 24;
pub const ACTION_COUNT: usize = 4;

const RAYS: [(i32, i32); 8] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyBrain {
    hidden_weights: Vec<f32>,
    hidden_bias: Vec<f32>,
    output_weights: Vec<f32>,
    output_bias: Vec<f32>,
}

impl PolicyBrain {
    pub fn random(rng: &mut SmallRng) -> Self {
        let hidden_weights = (0..HIDDEN_SIZE * OBSERVATION_SIZE)
            .map(|_| rng.gen_range(-1.0..1.0) * 0.35)
            .collect();
        let hidden_bias = (0..HIDDEN_SIZE)
            .map(|_| rng.gen_range(-1.0..1.0) * 0.15)
            .collect();
        let output_weights = (0..ACTION_COUNT * HIDDEN_SIZE)
            .map(|_| rng.gen_range(-1.0..1.0) * 0.35)
            .collect();
        let output_bias = (0..ACTION_COUNT)
            .map(|_| rng.gen_range(-1.0..1.0) * 0.15)
            .collect();

        Self {
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
        }
    }

    pub fn parameter_count() -> usize {
        (HIDDEN_SIZE * OBSERVATION_SIZE) + HIDDEN_SIZE + (ACTION_COUNT * HIDDEN_SIZE) + ACTION_COUNT
    }

    pub fn mutated_from(parent: &Self, rng: &mut SmallRng, sigma: f32) -> Self {
        let mut child = parent.clone();
        mutate_slice(&mut child.hidden_weights, rng, sigma);
        mutate_slice(&mut child.hidden_bias, rng, sigma * 0.7);
        mutate_slice(&mut child.output_weights, rng, sigma);
        mutate_slice(&mut child.output_bias, rng, sigma * 0.7);
        child
    }

    pub fn crossover_mutate(left: &Self, right: &Self, rng: &mut SmallRng, sigma: f32) -> Self {
        let mut child = Self {
            hidden_weights: mix_slice(&left.hidden_weights, &right.hidden_weights, rng),
            hidden_bias: mix_slice(&left.hidden_bias, &right.hidden_bias, rng),
            output_weights: mix_slice(&left.output_weights, &right.output_weights, rng),
            output_bias: mix_slice(&left.output_bias, &right.output_bias, rng),
        };

        mutate_slice(&mut child.hidden_weights, rng, sigma);
        mutate_slice(&mut child.hidden_bias, rng, sigma * 0.7);
        mutate_slice(&mut child.output_weights, rng, sigma);
        mutate_slice(&mut child.output_bias, rng, sigma * 0.7);
        child
    }

    pub fn decide(&self, state: &GameState) -> Direction {
        let observation = observe(state);
        let logits = self.forward(&observation);
        let ranked = rank_actions(logits);

        for action in ranked {
            let direction = index_to_direction(action);
            if direction == state.direction.opposite() {
                continue;
            }
            if !would_collide(state, direction) {
                return direction;
            }
        }

        for action in 0..ACTION_COUNT {
            let direction = index_to_direction(action);
            if direction != state.direction.opposite() {
                return direction;
            }
        }

        state.direction
    }

    fn forward(&self, observation: &[f32]) -> [f32; ACTION_COUNT] {
        let mut hidden = [0.0; HIDDEN_SIZE];
        for (hidden_index, value) in hidden.iter_mut().enumerate() {
            let row_offset = hidden_index * OBSERVATION_SIZE;
            let sum = observation
                .iter()
                .enumerate()
                .fold(self.hidden_bias[hidden_index], |acc, (obs_index, obs)| {
                    acc + self.hidden_weights[row_offset + obs_index] * obs
                });
            *value = leaky_relu(sum);
        }

        let mut output = [0.0; ACTION_COUNT];
        for (action_index, value) in output.iter_mut().enumerate() {
            let row_offset = action_index * HIDDEN_SIZE;
            let sum = hidden.iter().enumerate().fold(
                self.output_bias[action_index],
                |acc, (hidden_index, hidden_value)| {
                    acc + self.output_weights[row_offset + hidden_index] * hidden_value
                },
            );
            *value = sum;
        }

        output
    }
}

pub fn observe(state: &GameState) -> [f32; OBSERVATION_SIZE] {
    let mut observation = [0.0; OBSERVATION_SIZE];
    let head = state.head();
    let mut cursor = 0;

    for (dx, dy) in RAYS {
        let (food_seen, body_distance, wall_distance) = ray_features(state, head, dx, dy);
        observation[cursor] = food_seen;
        observation[cursor + 1] = body_distance;
        observation[cursor + 2] = wall_distance;
        cursor += 3;
    }

    observation[cursor] = (state.food.x - head.x) as f32 / state.grid_size as f32;
    observation[cursor + 1] = (state.food.y - head.y) as f32 / state.grid_size as f32;
    cursor += 2;

    observation[cursor] = if state.direction == Direction::Up {
        1.0
    } else {
        0.0
    };
    observation[cursor + 1] = if state.direction == Direction::Down {
        1.0
    } else {
        0.0
    };
    observation[cursor + 2] = if state.direction == Direction::Left {
        1.0
    } else {
        0.0
    };
    observation[cursor + 3] = if state.direction == Direction::Right {
        1.0
    } else {
        0.0
    };
    cursor += 4;

    observation[cursor] = state.snake.len() as f32 / (state.grid_size * state.grid_size) as f32;
    observation[cursor + 1] = state.steps_since_food as f32 / state.starvation_limit() as f32;
    observation[cursor + 2] = safe_move_ratio(state);
    observation[cursor + 3] = distance_from_center(head, state.grid_size);

    observation
}

pub fn would_collide(state: &GameState, direction: Direction) -> bool {
    let next = next_pos(state.head(), direction);
    state.would_collide(next)
}

fn ray_features(state: &GameState, head: Pos, dx: i32, dy: i32) -> (f32, f32, f32) {
    let mut x = head.x;
    let mut y = head.y;
    let mut distance: f32 = 0.0;
    let mut food_seen = 0.0;
    let mut body_distance = 0.0;

    loop {
        x += dx;
        y += dy;
        distance += 1.0;

        if x < 0 || x >= state.grid_size || y < 0 || y >= state.grid_size {
            let wall_distance = 1.0 / distance.max(1.0);
            return (food_seen, body_distance, wall_distance);
        }

        let pos = Pos { x, y };
        if food_seen == 0.0 && pos == state.food {
            food_seen = 1.0;
        }
        if body_distance == 0.0 && state.snake.iter().skip(1).any(|segment| *segment == pos) {
            body_distance = 1.0 / distance;
        }
    }
}

fn safe_move_ratio(state: &GameState) -> f32 {
    let safe = [
        Direction::Up,
        Direction::Down,
        Direction::Left,
        Direction::Right,
    ]
    .iter()
    .filter(|direction| !would_collide(state, **direction))
    .count();
    safe as f32 / ACTION_COUNT as f32
}

fn distance_from_center(pos: Pos, grid_size: i32) -> f32 {
    let center = (grid_size - 1) as f32 / 2.0;
    let dx = (pos.x as f32 - center).abs() / center.max(1.0);
    let dy = (pos.y as f32 - center).abs() / center.max(1.0);
    ((dx + dy) * 0.5).min(1.0)
}

fn next_pos(head: Pos, direction: Direction) -> Pos {
    match direction {
        Direction::Up => Pos {
            x: head.x,
            y: head.y - 1,
        },
        Direction::Down => Pos {
            x: head.x,
            y: head.y + 1,
        },
        Direction::Left => Pos {
            x: head.x - 1,
            y: head.y,
        },
        Direction::Right => Pos {
            x: head.x + 1,
            y: head.y,
        },
    }
}

fn rank_actions(logits: [f32; ACTION_COUNT]) -> [usize; ACTION_COUNT] {
    let mut indices = [0, 1, 2, 3];
    indices.sort_by(|left, right| logits[*right].partial_cmp(&logits[*left]).unwrap());
    indices
}

fn index_to_direction(index: usize) -> Direction {
    match index {
        0 => Direction::Up,
        1 => Direction::Down,
        2 => Direction::Left,
        _ => Direction::Right,
    }
}

fn mix_slice(left: &[f32], right: &[f32], rng: &mut SmallRng) -> Vec<f32> {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| if rng.gen_bool(0.5) { *lhs } else { *rhs })
        .collect()
}

fn mutate_slice(values: &mut [f32], rng: &mut SmallRng, sigma: f32) {
    for value in values {
        if rng.gen_bool(0.92) {
            continue;
        }
        *value += rng.gen_range(-sigma..sigma);
    }
}

fn leaky_relu(value: f32) -> f32 {
    if value > 0.0 {
        value
    } else {
        value * 0.08
    }
}
