use crate::ai::PolicyBrain;
use crate::game::{Direction, GameState};
use crate::swarm::GRID_SIZE;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

const REPLAY_SEED: u64 = 20260408;
const MAX_REPLAY_TICKS: u32 = 260;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayTrace {
    pub seed: u64,
    pub steps: Vec<Direction>,
    pub final_score: u32,
    pub final_foods: u32,
    pub total_steps: u32,
}

pub fn record_replay(brain: &PolicyBrain) -> ReplayTrace {
    let mut rng = SmallRng::seed_from_u64(REPLAY_SEED);
    let mut state = GameState::new(GRID_SIZE, &mut rng);
    let mut steps = Vec::new();

    for _ in 0..MAX_REPLAY_TICKS {
        if state.is_dead {
            break;
        }

        let direction = brain.decide(&state);
        steps.push(direction);
        let _ = state.step(direction, &mut rng);
        if state.is_dead {
            break;
        }
    }

    ReplayTrace {
        seed: REPLAY_SEED,
        steps,
        final_score: state.score,
        final_foods: state.foods_eaten,
        total_steps: state.steps_alive,
    }
}

pub fn replay_frames(trace: &ReplayTrace) -> Vec<GameState> {
    let mut rng = SmallRng::seed_from_u64(trace.seed);
    let mut state = GameState::new(GRID_SIZE, &mut rng);
    let mut frames = vec![state.clone()];

    for direction in &trace.steps {
        if state.is_dead {
            break;
        }
        let _ = state.step(*direction, &mut rng);
        frames.push(state.clone());
    }

    frames
}
