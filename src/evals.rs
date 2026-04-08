use crate::ai::{would_collide, PolicyBrain};
use crate::game::{Direction, GameState, Pos, StepOutcome};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::swarm::GRID_SIZE;

const EVAL_MAX_TICKS: u32 = 260;
const EVAL_SEEDS: [u64; 12] = [
    11, 23, 37, 41, 53, 67, 79, 97, 113, 131, 149, 167,
];

#[derive(Debug, Clone, PartialEq)]
pub struct EvalSummary {
    pub runs: usize,
    pub mean_score: f32,
    pub mean_foods: f32,
    pub mean_fitness: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompareReport {
    pub seeds: usize,
    pub current: EvalSummary,
    pub selected: EvalSummary,
}

pub fn compare_policies(current: Option<&PolicyBrain>, selected: Option<&PolicyBrain>) -> CompareReport {
    CompareReport {
        seeds: EVAL_SEEDS.len(),
        current: evaluate_policy(current, &EVAL_SEEDS),
        selected: evaluate_policy(selected, &EVAL_SEEDS),
    }
}

fn evaluate_policy(policy: Option<&PolicyBrain>, seeds: &[u64]) -> EvalSummary {
    let mut total_score = 0.0;
    let mut total_foods = 0.0;
    let mut total_fitness = 0.0;

    for seed in seeds {
        let mut rng = SmallRng::seed_from_u64(*seed);
        let mut state = GameState::new(GRID_SIZE, &mut rng);
        let mut fitness = 0.0;

        for _ in 0..EVAL_MAX_TICKS {
            if state.is_dead {
                break;
            }

            let direction = match policy {
                Some(policy) => policy.decide(&state),
                None => fallback_direction(&state),
            };
            let outcome = state.step(direction, &mut rng);
            fitness += shaped_reward(&state, outcome);

            if state.is_dead {
                break;
            }
        }

        total_score += state.score as f32;
        total_foods += state.foods_eaten as f32;
        total_fitness += fitness;
    }

    let runs = seeds.len().max(1);
    EvalSummary {
        runs: seeds.len(),
        mean_score: total_score / runs as f32,
        mean_foods: total_foods / runs as f32,
        mean_fitness: total_fitness / runs as f32,
    }
}

fn fallback_direction(state: &GameState) -> Direction {
    let mut best_direction = state.direction;
    let mut best_distance = i32::MAX;

    for direction in [
        Direction::Up,
        Direction::Down,
        Direction::Left,
        Direction::Right,
    ] {
        if direction == state.direction.opposite() || would_collide(state, direction) {
            continue;
        }

        let next = next_pos(state.head(), direction);
        let distance = (next.x - state.food.x).abs() + (next.y - state.food.y).abs();
        if distance < best_distance {
            best_distance = distance;
            best_direction = direction;
        }
    }

    if best_distance != i32::MAX {
        return best_direction;
    }

    for direction in [
        Direction::Up,
        Direction::Down,
        Direction::Left,
        Direction::Right,
    ] {
        if direction != state.direction.opposite() {
            return direction;
        }
    }

    state.direction
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

fn shaped_reward(state: &GameState, outcome: StepOutcome) -> f32 {
    let mut reward = -0.015;

    match outcome {
        StepOutcome::AteFood => reward += 26.0,
        StepOutcome::Closer(delta) => reward += delta as f32 * 0.42,
        StepOutcome::Further(delta) => reward -= delta as f32 * 0.22,
        StepOutcome::Neutral => reward += 0.03,
        StepOutcome::Dead => reward -= 18.0,
    }

    reward += state.foods_eaten as f32 * 0.8;
    reward += state.steps_alive as f32 * 0.01;
    reward
}
