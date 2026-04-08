use crate::ai::PolicyBrain;
use crate::game::{ArenaConfig, GameState, StepOutcome};
use crate::persistence::save_generation_checkpoint;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

pub const GRID_SIZE: i32 = 20;
pub const POPULATION_SIZE: usize = 1024;
pub const SAMPLE_BOARD_COUNT: usize = 144;
pub const POPULATION_MATRIX_COLUMNS: usize = 32;
pub const HEATMAP_CELLS: usize = (GRID_SIZE as usize) * (GRID_SIZE as usize);

const HARD_ELITES: usize = 16;
const MAX_TICKS_PER_GENERATION: u32 = 260;
const SNAPSHOT_INTERVAL: u32 = 4;
const SIMULATION_SLEEP_MS: u64 = 18;
const INITIAL_MUTATION_SCALE: f32 = 0.18;
const CURRICULUM_STAGE_THRESHOLDS: [u32; 3] = [0, 80, 120];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SwarmStats {
    pub generation: u32,
    pub population_size: u32,
    pub alive_agents: u32,
    #[serde(default = "default_arena_stage")]
    pub arena_stage: String,
    pub champion_score: u32,
    pub champion_foods: u32,
    pub champion_fitness: f32,
    pub mean_fitness: f32,
    pub median_fitness: f32,
    pub mutation_scale: f32,
    pub checkpoints_saved: u32,
    pub total_agent_steps: u64,
    pub total_foods: u64,
    pub steps_per_second: f32,
}

impl Default for SwarmStats {
    fn default() -> Self {
        Self {
            generation: 0,
            population_size: POPULATION_SIZE as u32,
            alive_agents: POPULATION_SIZE as u32,
            arena_stage: default_arena_stage(),
            champion_score: 0,
            champion_foods: 0,
            champion_fitness: 0.0,
            mean_fitness: 0.0,
            median_fitness: 0.0,
            mutation_scale: INITIAL_MUTATION_SCALE,
            checkpoints_saved: 0,
            total_agent_steps: 0,
            total_foods: 0,
            steps_per_second: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PopulationCell {
    pub id: usize,
    pub alive: bool,
    pub score: u32,
    pub foods: u32,
    pub fitness: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampleAgent {
    pub id: usize,
    pub state: GameState,
    pub fitness: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChampionView {
    pub state: GameState,
    pub fitness: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwarmSnapshot {
    pub stats: SwarmStats,
    pub population_cells: Vec<PopulationCell>,
    pub sample_agents: Vec<SampleAgent>,
    pub head_heatmap: Vec<u32>,
    pub champion: Option<ChampionView>,
}

impl SwarmSnapshot {
    pub fn empty(stats: SwarmStats) -> Self {
        let population_cells = (0..POPULATION_SIZE)
            .map(|id| PopulationCell {
                id,
                alive: false,
                score: 0,
                foods: 0,
                fitness: 0.0,
            })
            .collect();

        Self {
            stats,
            population_cells,
            sample_agents: Vec::new(),
            head_heatmap: vec![0; HEATMAP_CELLS],
            champion: None,
        }
    }
}

#[derive(Clone)]
struct Candidate {
    id: usize,
    brain: PolicyBrain,
    state: GameState,
    fitness: f32,
}

fn default_arena_stage() -> String {
    "Standard Arena".to_string()
}

impl Candidate {
    fn new(id: usize, brain: PolicyBrain, arena: &ArenaConfig, rng: &mut SmallRng) -> Self {
        Self {
            id,
            brain,
            state: GameState::new_with_arena(arena.clone(), rng),
            fitness: 0.0,
        }
    }
}

pub fn run_swarm(
    shared_brain: Arc<RwLock<Option<PolicyBrain>>>,
    shared_stats: Arc<RwLock<SwarmStats>>,
    shared_snapshot: Arc<RwLock<SwarmSnapshot>>,
    reset_signal: Arc<AtomicU64>,
) {
    let mut rng = SmallRng::from_entropy();
    let mut mutation_scale = {
        let stats = shared_stats.read().unwrap();
        if stats.mutation_scale > 0.0 {
            stats.mutation_scale
        } else {
            INITIAL_MUTATION_SCALE
        }
    };

    let seed_brain = shared_brain.read().unwrap().clone();
    let mut generation = shared_stats.read().unwrap().generation;
    let mut total_steps = shared_stats.read().unwrap().total_agent_steps;
    let mut total_foods = shared_stats.read().unwrap().total_foods;
    let mut checkpoints_saved = shared_stats.read().unwrap().checkpoints_saved;
    let mut last_reset_seen = reset_signal.load(Ordering::Relaxed);
    let mut previous_best = shared_stats.read().unwrap().champion_fitness;
    let mut stage_index = arena_stage_index(&shared_stats.read().unwrap().arena_stage);
    let mut arena = ArenaConfig::curriculum_stage(stage_index, GRID_SIZE);
    let mut population = seed_population(seed_brain.as_ref(), &arena, mutation_scale, &mut rng);
    let mut champion_view = None;

    loop {
        let reset_now = reset_signal.load(Ordering::Relaxed);
        if reset_now != last_reset_seen {
            generation = 0;
            total_steps = 0;
            total_foods = 0;
            checkpoints_saved = 0;
            previous_best = 0.0;
            mutation_scale = INITIAL_MUTATION_SCALE;
            champion_view = None;
            stage_index = 0;
            arena = ArenaConfig::curriculum_stage(stage_index, GRID_SIZE);
            let reseed_brain = shared_brain.read().unwrap().clone();
            population = seed_population(reseed_brain.as_ref(), &arena, mutation_scale, &mut rng);
            last_reset_seen = reset_now;
            write_shared_stats(
                &shared_stats,
                SwarmStats {
                    arena_stage: arena.stage_label().to_string(),
                    mutation_scale,
                    ..SwarmStats::default()
                },
            );
            write_shared_snapshot(
                &shared_snapshot,
                SwarmSnapshot::empty(SwarmStats {
                    arena_stage: arena.stage_label().to_string(),
                    ..SwarmStats::default()
                }),
            );
        }

        let evaluation_start = Instant::now();
        let generation_step_start = total_steps;
        let mut tick = 0;
        loop {
            let mut alive_agents = 0usize;
            let mut head_heatmap = vec![0u32; HEATMAP_CELLS];

            for candidate in &mut population {
                if candidate.state.is_dead {
                    continue;
                }

                alive_agents += 1;
                let direction = candidate.brain.decide(&candidate.state);
                let outcome = candidate.state.step(direction, &mut rng);
                total_steps += 1;
                candidate.fitness += shaped_reward(&candidate.state, outcome);
                if outcome == StepOutcome::AteFood {
                    total_foods += 1;
                }

                let head = candidate.state.head();
                let index = (head.y * GRID_SIZE + head.x) as usize;
                if index < head_heatmap.len() {
                    head_heatmap[index] += 1;
                }
            }

            if tick % SNAPSHOT_INTERVAL == 0
                || alive_agents == 0
                || tick + 1 == MAX_TICKS_PER_GENERATION
            {
                let snapshot = snapshot_from_population(
                    &population,
                    alive_agents,
                    arena.stage_label(),
                    mutation_scale,
                    generation,
                    total_steps - generation_step_start,
                    total_steps,
                    total_foods,
                    evaluation_start.elapsed(),
                    champion_view.clone(),
                    head_heatmap,
                );
                write_shared_snapshot(&shared_snapshot, snapshot.clone());
                write_shared_stats(&shared_stats, snapshot.stats.clone());
            }

            tick += 1;
            if alive_agents == 0 || tick >= MAX_TICKS_PER_GENERATION {
                break;
            }
            std::thread::sleep(Duration::from_millis(SIMULATION_SLEEP_MS));
        }

        population.sort_by(|left, right| right.fitness.partial_cmp(&left.fitness).unwrap());
        let champion = population[0].clone();
        generation += 1;

        let mean_fitness = population
            .iter()
            .map(|candidate| candidate.fitness)
            .sum::<f32>()
            / POPULATION_SIZE as f32;
        let median_fitness = population[POPULATION_SIZE / 2].fitness;
        let generation_steps = total_steps - generation_step_start;
        let steps_per_second =
            generation_steps as f32 / evaluation_start.elapsed().as_secs_f32().max(0.001);

        let improved = champion.fitness > previous_best;
        if improved {
            mutation_scale = (mutation_scale * 0.95).max(0.05);
            previous_best = champion.fitness;
        } else {
            mutation_scale = (mutation_scale * 1.03).min(0.35);
        }

        champion_view = Some(ChampionView {
            state: champion.state.clone(),
            fitness: champion.fitness,
        });

        if stage_index + 1 < CURRICULUM_STAGE_THRESHOLDS.len()
            && champion.state.score >= CURRICULUM_STAGE_THRESHOLDS[stage_index + 1]
        {
            stage_index += 1;
            arena = ArenaConfig::curriculum_stage(stage_index, GRID_SIZE);
        }

        let mut stats = SwarmStats {
            generation,
            population_size: POPULATION_SIZE as u32,
            alive_agents: 0,
            arena_stage: arena.stage_label().to_string(),
            champion_score: champion.state.score,
            champion_foods: champion.state.foods_eaten,
            champion_fitness: champion.fitness,
            mean_fitness,
            median_fitness,
            mutation_scale,
            checkpoints_saved,
            total_agent_steps: total_steps,
            total_foods,
            steps_per_second,
        };

        match save_generation_checkpoint(&champion.brain, &stats) {
            Ok(Some(_)) => {
                checkpoints_saved += 1;
                stats.checkpoints_saved = checkpoints_saved;
            }
            Ok(None) => {}
            Err(error) => {
                eprintln!("checkpoint save failed: {error}");
            }
        }

        write_shared_brain(&shared_brain, Some(champion.brain.clone()));
        let final_snapshot = snapshot_after_generation(&population, &stats, champion_view.clone());
        write_shared_stats(&shared_stats, stats.clone());
        write_shared_snapshot(&shared_snapshot, final_snapshot);

        population = reseed_population_from_champion(&champion.brain, &arena, mutation_scale, &mut rng);
    }
}


fn seed_population(
    seed: Option<&PolicyBrain>,
    arena: &ArenaConfig,
    sigma: f32,
    rng: &mut SmallRng,
) -> Vec<Candidate> {
    (0..POPULATION_SIZE)
        .map(|id| {
            let brain = match seed {
                Some(brain) if id == 0 => brain.clone(),
                Some(brain) if id < HARD_ELITES * 2 => PolicyBrain::mutated_from(brain, rng, sigma),
                Some(brain) if id < POPULATION_SIZE / 2 => {
                    PolicyBrain::mutated_from(brain, rng, sigma * 1.4)
                }
                _ => PolicyBrain::random(rng),
            };
            Candidate::new(id, brain, arena, rng)
        })
        .collect()
}

fn reseed_population_from_champion(
    champion: &PolicyBrain,
    arena: &ArenaConfig,
    sigma: f32,
    rng: &mut SmallRng,
) -> Vec<Candidate> {
    // The next generation is rebuilt around the strongest live policy instead
    // of mixing survivors from the old pool. This keeps the whole lab focused
    // on the latest best model and clears weaker branches immediately.
    seed_population(Some(champion), arena, sigma, rng)
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

fn snapshot_from_population(
    population: &[Candidate],
    alive_agents: usize,
    arena_stage: &str,
    mutation_scale: f32,
    generation: u32,
    generation_steps: u64,
    total_steps: u64,
    total_foods: u64,
    elapsed: Duration,
    champion: Option<ChampionView>,
    head_heatmap: Vec<u32>,
) -> SwarmSnapshot {
    let population_cells = population
        .iter()
        .map(|candidate| PopulationCell {
            id: candidate.id,
            alive: candidate.state.is_dead == false,
            score: candidate.state.score,
            foods: candidate.state.foods_eaten,
            fitness: candidate.fitness,
        })
        .collect::<Vec<_>>();

    let sample_agents = sample_agents(population);
    let champion_score = population
        .iter()
        .map(|candidate| candidate.state.score)
        .max()
        .unwrap_or(0);
    let champion_foods = population
        .iter()
        .map(|candidate| candidate.state.foods_eaten)
        .max()
        .unwrap_or(0);
    let mean_fitness = population
        .iter()
        .map(|candidate| candidate.fitness)
        .sum::<f32>()
        / POPULATION_SIZE as f32;
    let mut sorted_fitness = population
        .iter()
        .map(|candidate| candidate.fitness)
        .collect::<Vec<_>>();
    sorted_fitness.sort_by(|left, right| right.partial_cmp(left).unwrap());
    let median_fitness = sorted_fitness[sorted_fitness.len() / 2];

    SwarmSnapshot {
        stats: SwarmStats {
            generation,
            population_size: POPULATION_SIZE as u32,
            alive_agents: alive_agents as u32,
            arena_stage: arena_stage.to_string(),
            champion_score,
            champion_foods,
            champion_fitness: sorted_fitness[0],
            mean_fitness,
            median_fitness,
            mutation_scale,
            checkpoints_saved: 0,
            total_agent_steps: total_steps,
            total_foods,
            steps_per_second: generation_steps as f32 / elapsed.as_secs_f32().max(0.001),
        },
        population_cells,
        sample_agents,
        head_heatmap,
        champion,
    }
}

fn snapshot_after_generation(
    population: &[Candidate],
    stats: &SwarmStats,
    champion: Option<ChampionView>,
) -> SwarmSnapshot {
    let mut heatmap = vec![0u32; HEATMAP_CELLS];
    for candidate in population.iter().take(POPULATION_SIZE.min(128)) {
        let head = candidate.state.head();
        let index = (head.y * GRID_SIZE + head.x) as usize;
        if index < heatmap.len() {
            heatmap[index] += 1;
        }
    }

    SwarmSnapshot {
        stats: stats.clone(),
        population_cells: population
            .iter()
            .map(|candidate| PopulationCell {
                id: candidate.id,
                alive: candidate.state.is_dead == false,
                score: candidate.state.score,
                foods: candidate.state.foods_eaten,
                fitness: candidate.fitness,
            })
            .collect(),
        sample_agents: sample_agents(population),
        head_heatmap: heatmap,
        champion,
    }
}

fn arena_stage_index(label: &str) -> usize {
    match label {
        "Sparse Food Arena" => 1,
        "Obstacle Field" => 2,
        _ => 0,
    }
}

fn sample_agents(population: &[Candidate]) -> Vec<SampleAgent> {
    if population.is_empty() {
        return Vec::new();
    }

    let stride = (population.len() / SAMPLE_BOARD_COUNT.max(1)).max(1);
    population
        .iter()
        .step_by(stride)
        .take(SAMPLE_BOARD_COUNT)
        .map(|candidate| SampleAgent {
            id: candidate.id,
            state: candidate.state.clone(),
            fitness: candidate.fitness,
        })
        .collect()
}

fn write_shared_brain(shared_brain: &Arc<RwLock<Option<PolicyBrain>>>, brain: Option<PolicyBrain>) {
    let mut guard = shared_brain.write().unwrap();
    *guard = brain;
}

fn write_shared_stats(shared_stats: &Arc<RwLock<SwarmStats>>, stats: SwarmStats) {
    let mut guard = shared_stats.write().unwrap();
    *guard = stats;
}

fn write_shared_snapshot(shared_snapshot: &Arc<RwLock<SwarmSnapshot>>, snapshot: SwarmSnapshot) {
    let mut guard = shared_snapshot.write().unwrap();
    *guard = snapshot;
}
