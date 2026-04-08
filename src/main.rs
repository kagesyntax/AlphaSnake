mod ai;
mod evals;
mod game;
mod persistence;
mod replay;
mod swarm;

use crate::ai::{would_collide, PolicyBrain};
use crate::evals::CompareReport;
use crate::game::{Direction, GameState, Pos};
use crate::persistence::{
    archive_dir_display, artifacts_root_display, current_checkpoint_dir_display,
    current_checkpoint_info, current_policy_path_display, list_checkpoints, load_checkpoint_bundle,
    load_current, load_replay, promote_checkpoint, save_current, CheckpointInfo,
};
use crate::replay::{replay_frames, ReplayTrace};
use crate::swarm::{run_swarm, PopulationCell, SwarmSnapshot, SwarmStats};
use dioxus::prelude::*;
use futures_timer::Delay;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

const GRID_SIZE: i32 = 20;
const PLAYER_CELL_SIZE: i32 = 20;
const OPPONENT_CELL_SIZE: i32 = 18;
const CHAMPION_CELL_SIZE: i32 = 12;
const REPLAY_CELL_SIZE: i32 = 12;
const SAMPLE_CELL_SIZE: i32 = 6;

fn main() {
    if run_benchmark_cli() {
        return;
    }
    dioxus::launch(App);
}

fn run_benchmark_cli() -> bool {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) != Some("benchmark") {
        return false;
    }

    if args.len() != 3 {
        eprintln!("usage: cargo run -- benchmark <current|checkpoint-dir> <current|checkpoint-dir>");
        return true;
    }

    let left = load_policy_arg(&args[1]);
    let right = load_policy_arg(&args[2]);
    let report = crate::evals::compare_policies(left.as_ref(), right.as_ref());

    println!("benchmark seeds: {}", report.seeds);
    println!(
        "left  mean_score={:.1} mean_foods={:.2} mean_fitness={:.2}",
        report.current.mean_score, report.current.mean_foods, report.current.mean_fitness
    );
    println!(
        "right mean_score={:.1} mean_foods={:.2} mean_fitness={:.2}",
        report.selected.mean_score, report.selected.mean_foods, report.selected.mean_fitness
    );

    true
}

fn load_policy_arg(argument: &str) -> Option<PolicyBrain> {
    if argument == "current" {
        return load_current().brain;
    }

    match load_checkpoint_bundle(argument) {
        Ok(bundle) => bundle.brain,
        Err(error) => {
            eprintln!("failed to load checkpoint `{argument}`: {error}");
            None
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Game,
    Lab,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AlertLevel {
    Info,
    Success,
    Warning,
}

#[derive(Clone, PartialEq)]
struct LiveAlert {
    id: u64,
    level: AlertLevel,
    headline: String,
    detail: String,
}

#[derive(Clone, PartialEq)]
struct LoadedReplay {
    source: CheckpointInfo,
    trace: ReplayTrace,
    frames: Vec<GameState>,
}

#[component]
fn App() -> Element {
    let persisted = use_memo(load_current);
    let initial = persisted();
    let initial_brain = initial.brain.clone();
    let initial_stats = initial.stats.clone();
    let initial_stats_for_store = initial_stats.clone();
    let initial_stats_for_snapshot = initial_stats.clone();

    let brain_store = use_signal(move || Arc::new(RwLock::new(initial_brain.clone())));
    let stats_store = use_signal(move || Arc::new(RwLock::new(initial_stats_for_store.clone())));
    let snapshot_store = use_signal(move || {
        Arc::new(RwLock::new(SwarmSnapshot::empty(
            initial_stats_for_snapshot.clone(),
        )))
    });
    let reset_signal = use_signal(|| Arc::new(AtomicU64::new(0)));

    let mut screen = use_signal(|| Screen::Game);
    let mut human_dir = use_signal(|| Direction::Up);
    let mut human_rng = use_signal(SmallRng::from_entropy);
    let mut ai_rng = use_signal(SmallRng::from_entropy);
    let mut human_game = use_signal(|| {
        let mut rng = SmallRng::from_entropy();
        GameState::new(GRID_SIZE, &mut rng)
    });
    let mut ai_game = use_signal(|| {
        let mut rng = SmallRng::from_entropy();
        GameState::new(GRID_SIZE, &mut rng)
    });
    let mut player_best_score = use_signal(|| 0_u32);
    let mut ai_best_score = use_signal(|| 0_u32);
    let mut operator_log = use_signal(|| "Research console online".to_string());
    let mut alerts = use_signal(Vec::<LiveAlert>::new);
    let mut next_alert_id = use_signal(|| 1_u64);
    let mut selected_checkpoint_dir = use_signal(|| None::<String>);
    let mut compare_report = use_signal(|| None::<CompareReport>);
    let mut loaded_replay = use_signal(|| None::<LoadedReplay>);
    let mut replay_index = use_signal(|| 0_usize);
    let mut replay_playing = use_signal(|| false);
    let mut seen_generation = use_signal(|| initial_stats.generation);
    let mut seen_champion_score = use_signal(|| initial_stats.champion_score);
    let mut seen_checkpoints = use_signal(|| initial_stats.checkpoints_saved);
    let mut refresh = use_signal(|| 0_u64);
    let mut swarm_started = use_signal(|| false);

    let shared_brain = brain_store();
    let shared_stats = stats_store();
    let shared_snapshot = snapshot_store();
    let shared_reset = reset_signal();

    let shared_brain_for_swarm = shared_brain.clone();
    let shared_stats_for_swarm = shared_stats.clone();
    let shared_snapshot_for_swarm = shared_snapshot.clone();
    let shared_reset_for_swarm = shared_reset.clone();
    let shared_reset_for_button = shared_reset.clone();

    use_effect(move || {
        if swarm_started() {
            return;
        }

        swarm_started.set(true);
        let brain = shared_brain_for_swarm.clone();
        let stats = shared_stats_for_swarm.clone();
        let snapshot = shared_snapshot_for_swarm.clone();
        let reset = shared_reset_for_swarm.clone();

        std::thread::spawn(move || {
            run_swarm(brain, stats, snapshot, reset);
        });
    });

    let brain_for_game = brain_store();
    use_future(move || {
        let brain_for_game = brain_for_game.clone();
        async move {
            loop {
                Delay::new(Duration::from_millis(120)).await;

                human_game.with_mut(|game| {
                    if game.is_dead {
                        return;
                    }
                    human_rng.with_mut(|rng| {
                        let _ = game.step(human_dir(), rng);
                    });
                });
                player_best_score.with_mut(|best| {
                    *best = (*best).max(human_game.read().score);
                });

                let current_brain = brain_for_game.read().unwrap().clone();
                ai_game.with_mut(move |game| {
                    ai_rng.with_mut(|rng| {
                        advance_ai_game(game, rng, current_brain.as_ref());
                    });
                });
                ai_best_score.with_mut(|best| {
                    *best = (*best).max(ai_game.read().score);
                });
            }
        }
    });

    use_future(move || async move {
        loop {
            Delay::new(Duration::from_millis(250)).await;
            refresh.with_mut(|tick| *tick += 1);
        }
    });

    use_future(move || async move {
        loop {
            Delay::new(Duration::from_millis(140)).await;
            if !replay_playing() {
                continue;
            }

            let frame_count = loaded_replay()
                .map(|replay| replay.frames.len())
                .unwrap_or(0);
            if frame_count <= 1 {
                replay_playing.set(false);
                continue;
            }

            replay_index.with_mut(|index| {
                if *index + 1 >= frame_count {
                    *index = frame_count - 1;
                    replay_playing.set(false);
                } else {
                    *index += 1;
                }
            });
        }
    });

    let shared_stats_for_alerts = shared_stats.clone();
    use_effect(move || {
        let _ = refresh();
        let stats = shared_stats_for_alerts.read().unwrap().clone();

        if stats.generation > seen_generation() {
            push_alert(
                &mut alerts,
                &mut next_alert_id,
                AlertLevel::Info,
                format!("Generation {}", stats.generation),
                format!(
                    "Swarm advanced with champion score {} and mean fitness {:.2}.",
                    stats.champion_score, stats.mean_fitness
                ),
            );
            seen_generation.set(stats.generation);
        }

        if stats.champion_score > seen_champion_score() {
            push_alert(
                &mut alerts,
                &mut next_alert_id,
                AlertLevel::Success,
                "New champion score".to_string(),
                format!(
                    "Champion score improved from {} to {}.",
                    seen_champion_score(),
                    stats.champion_score
                ),
            );
            seen_champion_score.set(stats.champion_score);
        }

        if stats.checkpoints_saved > seen_checkpoints() {
            push_alert(
                &mut alerts,
                &mut next_alert_id,
                AlertLevel::Info,
                "Checkpoint archived".to_string(),
                format!(
                    "Trainer archived checkpoint {} at generation {}.",
                    stats.checkpoints_saved,
                    stats.generation
                ),
            );
            seen_checkpoints.set(stats.checkpoints_saved);
        }
    });

    use_effect(move || {
        let _ = refresh();
        if selected_checkpoint_dir().is_some() {
            return;
        }

        if let Some(checkpoint) = list_checkpoints(1).into_iter().next() {
            selected_checkpoint_dir.set(Some(checkpoint.directory));
        }
    });

    let _ = refresh();
    let current_stats = shared_stats.read().unwrap().clone();
    let current_snapshot = shared_snapshot.read().unwrap().clone();
    let has_brain = shared_brain.read().unwrap().is_some();
    let checkpoints = list_checkpoints(12);
    let selected_checkpoint = checkpoints
        .iter()
        .find(|checkpoint| {
            selected_checkpoint_dir()
                .as_ref()
                .map(|directory| directory == &checkpoint.directory)
                .unwrap_or(false)
        })
        .cloned()
        .or_else(|| checkpoints.first().cloned());
    let current_checkpoint = current_checkpoint_info();
    let replay_session = loaded_replay();
    let replay_frame = replay_session
        .as_ref()
        .and_then(|replay| replay.frames.get(replay_index()).cloned());
    let lab_best_score = checkpoints
        .iter()
        .map(|checkpoint| checkpoint.champion_score)
        .max()
        .unwrap_or(current_stats.champion_score);
    let artifacts_root = absolute_path(&artifacts_root_display());
    let current_dir = absolute_path(&current_checkpoint_dir_display());
    let policy_path = absolute_path(&current_policy_path_display());
    let archive_dir = absolute_path(&archive_dir_display());
    let shared_brain_for_save = shared_brain.clone();
    let shared_stats_for_save = shared_stats.clone();
    let shared_brain_for_promote = shared_brain.clone();
    let shared_stats_for_promote = shared_stats.clone();
    let shared_brain_for_compare = shared_brain.clone();
    let selected_checkpoint_for_load = selected_checkpoint.clone();
    let selected_checkpoint_for_compare = selected_checkpoint.clone();
    let selected_checkpoint_for_replay = selected_checkpoint.clone();

    rsx! {
        div {
            class: "console-shell",
            tabindex: "0",
            autofocus: "true",
            onkeydown: move |evt| {
                match evt.key() {
                    Key::ArrowUp => {
                        evt.prevent_default();
                        human_dir.set(Direction::Up);
                    }
                    Key::ArrowDown => {
                        evt.prevent_default();
                        human_dir.set(Direction::Down);
                    }
                    Key::ArrowLeft => {
                        evt.prevent_default();
                        human_dir.set(Direction::Left);
                    }
                    Key::ArrowRight => {
                        evt.prevent_default();
                        human_dir.set(Direction::Right);
                    }
                    Key::Character(c) => match c.as_str() {
                        "w" | "W" => human_dir.set(Direction::Up),
                        "s" | "S" => human_dir.set(Direction::Down),
                        "a" | "A" => human_dir.set(Direction::Left),
                        "d" | "D" => human_dir.set(Direction::Right),
                        "r" | "R" => {
                            human_game.with_mut(|game| {
                                human_rng.with_mut(|rng| {
                                    *game = GameState::new(GRID_SIZE, rng);
                                });
                            });
                            operator_log.set("Player board reset".to_string());
                        }
                        _ => {}
                    },
                    _ => {}
                }
            },

            style { {include_str!("../assets/research.css")} }

            header {
                class: "panel app-header",

                div {
                    class: "app-header__top",

                    div {
                        h1 { class: "app-header__title", "Snake Systems Lab" }
                        p { class: "app-header__eyebrow", "Interactive arena, live trainer, archived checkpoints" }
                        p { class: "app-header__copy", "Use `Game` to play against the current best policy. Use `Lab` to inspect the background population and its saved checkpoints." }
                    }

                    div {
                        class: "view-nav",
                        NavButton {
                            label: "Game".to_string(),
                            active: screen() == Screen::Game,
                            onclick: move |_| screen.set(Screen::Game),
                        }
                        NavButton {
                            label: "Lab".to_string(),
                            active: screen() == Screen::Lab,
                            onclick: move |_| screen.set(Screen::Lab),
                        }
                    }
                }

                div {
                    class: "status-strip",
                    StatusChip { label: "Generation".to_string(), value: current_stats.generation.to_string() }
                    StatusChip { label: "Arena".to_string(), value: current_stats.arena_stage.clone() }
                    StatusChip { label: "Champion Score".to_string(), value: current_stats.champion_score.to_string() }
                    StatusChip { label: "Population".to_string(), value: current_stats.population_size.to_string() }
                    StatusChip { label: "Policy".to_string(), value: if has_brain { "Loaded".to_string() } else { "Seeding".to_string() } }
                }
            }

            if screen() == Screen::Game {
                GameView {
                    human_game: human_game(),
                    ai_game: ai_game(),
                    has_brain,
                    stats: current_stats.clone(),
                    player_best_score: player_best_score(),
                    ai_best_score: ai_best_score(),
                    current_dir: current_dir.clone(),
                    policy_path: policy_path.clone(),
                    archive_dir: archive_dir.clone(),
                    operator_log: operator_log(),
                    alerts: alerts(),
                    on_reset_player: move |_| {
                        human_game.with_mut(|game| {
                            human_rng.with_mut(|rng| {
                                *game = GameState::new(GRID_SIZE, rng);
                            });
                        });
                        player_best_score.with_mut(|best| {
                            *best = (*best).max(human_game.read().score);
                        });
                        operator_log.set("Player board reset".to_string());
                        push_alert(
                            &mut alerts,
                            &mut next_alert_id,
                            AlertLevel::Info,
                            "Player board reset".to_string(),
                            "Manual board state reinitialized.".to_string(),
                        );
                    },
                    on_save: move |_| {
                        let brain = shared_brain_for_save.read().unwrap().clone();
                        let stats = shared_stats_for_save.read().unwrap().clone();
                        match save_current(&brain, &stats) {
                            Ok(report) => {
                                if report.has_brain {
                                    operator_log.set(format!("Saved active policy to {}", absolute_path(&report.policy_path)));
                                    push_alert(
                                        &mut alerts,
                                        &mut next_alert_id,
                                        AlertLevel::Success,
                                        "Checkpoint saved".to_string(),
                                        format!("Active policy saved to {}.", absolute_path(&report.policy_path)),
                                    );
                                } else {
                                    operator_log.set(format!("Saved stats only to {}", absolute_path(&report.current_dir)));
                                    push_alert(
                                        &mut alerts,
                                        &mut next_alert_id,
                                        AlertLevel::Info,
                                        "Stats snapshot saved".to_string(),
                                        format!("Saved stats to {}.", absolute_path(&report.current_dir)),
                                    );
                                }
                            }
                            Err(error) => {
                                operator_log.set(format!("Save failed: {error}"));
                                push_alert(
                                    &mut alerts,
                                    &mut next_alert_id,
                                    AlertLevel::Warning,
                                    "Checkpoint save failed".to_string(),
                                    error.to_string(),
                                );
                            }
                        }
                    },
                    on_reset_trainer: move |_| {
                        shared_reset_for_button.fetch_add(1, Ordering::Relaxed);
                        operator_log.set("Trainer reset requested".to_string());
                        push_alert(
                            &mut alerts,
                            &mut next_alert_id,
                            AlertLevel::Warning,
                            "Trainer reset requested".to_string(),
                            "Swarm state will be reseeded from scratch.".to_string(),
                        );
                    },
                }
            } else {
                LabView {
                    stats: current_stats.clone(),
                    snapshot: current_snapshot.clone(),
                    checkpoints: checkpoints.clone(),
                    current_checkpoint,
                    selected_checkpoint: selected_checkpoint.clone(),
                    compare_report: compare_report(),
                    replay: replay_session.clone(),
                    replay_frame,
                    replay_index: replay_index(),
                    replay_playing: replay_playing(),
                    lab_best_score,
                    artifacts_root,
                    current_dir,
                    policy_path,
                    archive_dir,
                    alerts: alerts(),
                    on_select_checkpoint: move |directory| {
                        selected_checkpoint_dir.set(Some(directory));
                    },
                    on_load_checkpoint: move |_| {
                        let Some(selected) = selected_checkpoint_for_load.clone() else {
                            operator_log.set("No checkpoint selected".to_string());
                            return;
                        };

                        match promote_checkpoint(&selected.directory) {
                            Ok(report) => {
                                {
                                    let mut brain = shared_brain_for_promote.write().unwrap();
                                    *brain = report.data.brain.clone();
                                }
                                {
                                    let mut stats = shared_stats_for_promote.write().unwrap();
                                    *stats = report.data.stats.clone();
                                }
                                shared_reset_for_button.fetch_add(1, Ordering::Relaxed);
                                compare_report.set(None);
                                operator_log.set(format!(
                                    "Promoted checkpoint generation {} into current slot",
                                    report.source.generation
                                ));
                                push_alert(
                                    &mut alerts,
                                    &mut next_alert_id,
                                    AlertLevel::Success,
                                    "Checkpoint promoted".to_string(),
                                    format!(
                                        "Loaded generation {} from {}.",
                                        report.source.generation,
                                        absolute_path(&report.source.directory)
                                    ),
                                );
                            }
                            Err(error) => {
                                operator_log.set(format!("Checkpoint load failed: {error}"));
                                push_alert(
                                    &mut alerts,
                                    &mut next_alert_id,
                                    AlertLevel::Warning,
                                    "Checkpoint load failed".to_string(),
                                    error.to_string(),
                                );
                            }
                        }
                    },
                    on_compare_checkpoint: move |_| {
                        let Some(selected) = selected_checkpoint_for_compare.clone() else {
                            operator_log.set("No checkpoint selected".to_string());
                            return;
                        };

                        match load_checkpoint_bundle(&selected.directory) {
                            Ok(bundle) => {
                                let current_brain = shared_brain_for_compare.read().unwrap().clone();
                                let report = crate::evals::compare_policies(
                                    current_brain.as_ref(),
                                    bundle.brain.as_ref(),
                                );
                                compare_report.set(Some(report));
                                operator_log.set(format!(
                                    "Compared current slot against generation {}",
                                    bundle.info.generation
                                ));
                                push_alert(
                                    &mut alerts,
                                    &mut next_alert_id,
                                    AlertLevel::Info,
                                    "Checkpoint comparison complete".to_string(),
                                    format!(
                                        "Benchmarked current slot against generation {} over fixed seeds.",
                                        bundle.info.generation
                                    ),
                                );
                            }
                            Err(error) => {
                                operator_log.set(format!("Comparison failed: {error}"));
                                push_alert(
                                    &mut alerts,
                                    &mut next_alert_id,
                                    AlertLevel::Warning,
                                    "Checkpoint comparison failed".to_string(),
                                    error.to_string(),
                                );
                            }
                        }
                    },
                    on_load_replay: move |_| {
                        let Some(selected) = selected_checkpoint_for_replay.clone() else {
                            operator_log.set("No checkpoint selected".to_string());
                            return;
                        };

                        match load_replay(&selected.directory) {
                            Ok(trace) => {
                                let frames = replay_frames(&trace);
                                loaded_replay.set(Some(LoadedReplay {
                                    source: selected.clone(),
                                    trace,
                                    frames,
                                }));
                                replay_index.set(0);
                                replay_playing.set(false);
                                operator_log.set(format!(
                                    "Loaded replay for generation {}",
                                    selected.generation
                                ));
                                push_alert(
                                    &mut alerts,
                                    &mut next_alert_id,
                                    AlertLevel::Info,
                                    "Replay loaded".to_string(),
                                    format!(
                                        "Loaded deterministic replay for generation {}.",
                                        selected.generation
                                    ),
                                );
                            }
                            Err(error) => {
                                operator_log.set(format!("Replay load failed: {error}"));
                                push_alert(
                                    &mut alerts,
                                    &mut next_alert_id,
                                    AlertLevel::Warning,
                                    "Replay load failed".to_string(),
                                    error.to_string(),
                                );
                            }
                        }
                    },
                    on_replay_prev: move |_| {
                        replay_playing.set(false);
                        replay_index.with_mut(|index| {
                            if *index > 0 {
                                *index -= 1;
                            }
                        });
                    },
                    on_replay_next: move |_| {
                        replay_playing.set(false);
                        let frame_count = loaded_replay()
                            .map(|replay| replay.frames.len())
                            .unwrap_or(0);
                        replay_index.with_mut(|index| {
                            if *index + 1 < frame_count {
                                *index += 1;
                            }
                        });
                    },
                    on_toggle_replay: move |_| {
                        if loaded_replay().is_none() {
                            return;
                        }
                        replay_playing.set(!replay_playing());
                    },
                }
            }

            footer {
                class: "panel app-footer",
                div {
                    class: "footer-grid",
                    div {
                        class: "footer-block",
                        p { class: "footer-block__label", "Operator Log" }
                        p { class: "footer-block__value", "{operator_log}" }
                    }
                    div {
                        class: "footer-block",
                        p { class: "footer-block__label", "Controls" }
                        p { class: "footer-block__value", "Move with arrows or `WASD`. Reset your board with `R`. The AI board auto-resets after death." }
                    }
                }
            }
        }
    }
}

fn advance_ai_game(game: &mut GameState, rng: &mut SmallRng, brain: Option<&PolicyBrain>) {
    if game.is_dead {
        *game = GameState::new(GRID_SIZE, rng);
    }

    let next_dir = match brain {
        Some(brain) => brain.decide(game),
        None => fallback_direction(game),
    };

    let _ = game.step(next_dir, rng);

    if game.is_dead {
        *game = GameState::new(GRID_SIZE, rng);
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

fn absolute_path(relative_path: &str) -> String {
    std::env::current_dir()
        .map(|dir| dir.join(relative_path).display().to_string())
        .unwrap_or_else(|_| relative_path.to_string())
}

fn population_cell_modifier(cell: &PopulationCell) -> &'static str {
    if !cell.alive {
        "population-cell--dead"
    } else if cell.score >= 80 {
        "population-cell--hot"
    } else if cell.score >= 20 {
        "population-cell--warm"
    } else {
        "population-cell--live"
    }
}

fn heatmap_color(value: u32, max: u32) -> String {
    if max == 0 || value == 0 {
        return "#1c2022".to_string();
    }

    let intensity = value as f32 / max as f32;
    let r = (58.0 + intensity * 145.0).round() as u8;
    let g = (72.0 + intensity * 120.0).round() as u8;
    let b = (64.0 + intensity * 55.0).round() as u8;
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn segment_style(segment: &Pos, cell_size: i32, color: &str, opacity: f32) -> String {
    format!(
        "width: {cell_size}px; height: {cell_size}px; left: {}px; top: {}px; background: {color}; opacity: {opacity};",
        segment.x * cell_size,
        segment.y * cell_size
    )
}

fn push_alert(
    alerts: &mut Signal<Vec<LiveAlert>>,
    next_alert_id: &mut Signal<u64>,
    level: AlertLevel,
    headline: String,
    detail: String,
) {
    let id = next_alert_id();
    next_alert_id.set(id + 1);
    alerts.with_mut(|items| {
        items.insert(
            0,
            LiveAlert {
                id,
                level,
                headline,
                detail,
            },
        );
        items.truncate(8);
    });
}

fn alert_level_label(level: AlertLevel) -> &'static str {
    match level {
        AlertLevel::Info => "Info",
        AlertLevel::Success => "Success",
        AlertLevel::Warning => "Warning",
    }
}

fn alert_level_class(level: AlertLevel) -> &'static str {
    match level {
        AlertLevel::Info => "alert-card alert-card--info",
        AlertLevel::Success => "alert-card alert-card--success",
        AlertLevel::Warning => "alert-card alert-card--warning",
    }
}

#[component]
fn NavButton(label: String, active: bool, onclick: EventHandler<MouseEvent>) -> Element {
    let class = if active {
        "view-button view-button--active"
    } else {
        "view-button"
    };

    rsx! {
        button {
            class,
            onclick: move |event| onclick.call(event),
            "{label}"
        }
    }
}

#[component]
fn StatusChip(label: String, value: String) -> Element {
    rsx! {
        div {
            class: "status-chip",
            p { class: "status-chip__label", "{label}" }
            p { class: "status-chip__value", "{value}" }
        }
    }
}

#[component]
fn MetricCard(label: String, value: String, detail: String) -> Element {
    rsx! {
        div {
            class: "telemetry-card",
            p { class: "telemetry-card__label", "{label}" }
            p { class: "telemetry-card__value", "{value}" }
            p { class: "telemetry-card__detail", "{detail}" }
        }
    }
}

#[component]
fn PathItem(label: String, value: String) -> Element {
    rsx! {
        div {
            class: "path-item",
            p { class: "path-item__label", "{label}" }
            p { class: "path-item__value", "{value}" }
        }
    }
}

#[component]
fn CheckpointMetaCard(title: String, checkpoint: Option<CheckpointInfo>) -> Element {
    rsx! {
        div {
            class: "checkpoint-meta",
            p { class: "path-item__label", "{title}" }
            if let Some(checkpoint) = checkpoint {
                p { class: "checkpoint-meta__value", "Gen {checkpoint.generation} / Score {checkpoint.champion_score}" }
                p { class: "checkpoint-meta__detail", "{checkpoint.arena_stage}" }
                p { class: "checkpoint-meta__detail", {format!("Fitness {:.2}", checkpoint.champion_fitness)} }
                p { class: "checkpoint-meta__detail", "{checkpoint.saved_at}" }
                p { class: "checkpoint-meta__path", "{absolute_path(&checkpoint.directory)}" }
            } else {
                p { class: "checkpoint-meta__detail", "No checkpoint available." }
            }
        }
    }
}

#[component]
fn CompareSummaryCard(report: CompareReport) -> Element {
    rsx! {
        div {
            class: "checkpoint-compare",
            p { class: "path-item__label", "Deterministic Compare" }
            p { class: "checkpoint-meta__detail", "Fixed seed pack: {report.seeds} runs per model." }
            div {
                class: "compare-grid",
                div {
                    class: "checkpoint-meta",
                    p { class: "checkpoint-meta__value", "Current" }
                    p { class: "checkpoint-meta__detail", {format!("Mean score {:.1}", report.current.mean_score)} }
                    p { class: "checkpoint-meta__detail", {format!("Mean foods {:.2}", report.current.mean_foods)} }
                    p { class: "checkpoint-meta__detail", {format!("Mean fitness {:.2}", report.current.mean_fitness)} }
                }
                div {
                    class: "checkpoint-meta",
                    p { class: "checkpoint-meta__value", "Selected" }
                    p { class: "checkpoint-meta__detail", {format!("Mean score {:.1}", report.selected.mean_score)} }
                    p { class: "checkpoint-meta__detail", {format!("Mean foods {:.2}", report.selected.mean_foods)} }
                    p { class: "checkpoint-meta__detail", {format!("Mean fitness {:.2}", report.selected.mean_fitness)} }
                }
            }
        }
    }
}

#[component]
fn ArchiveCheckpointEntry(
    checkpoint: CheckpointInfo,
    selected: bool,
    on_select_checkpoint: EventHandler<String>,
) -> Element {
    let class = if selected {
        "archive-entry archive-entry--selected"
    } else {
        "archive-entry"
    };

    rsx! {
        div {
            class,
            div {
                class: "archive-entry__top",
                span { "Gen {checkpoint.generation}" }
                span { "Score {checkpoint.champion_score}" }
            }
            p { class: "archive-entry__line", "{checkpoint.arena_stage}" }
            p { class: "archive-entry__line", {format!("Fitness {:.2}", checkpoint.champion_fitness)} }
            p { class: "archive-entry__line", "{checkpoint.saved_at}" }
            p { class: "archive-entry__path", "{absolute_path(&checkpoint.directory)}" }
            div {
                class: "archive-entry__actions",
                button {
                    class: "mini-button",
                    onclick: {
                        let directory = checkpoint.directory.clone();
                        move |_| on_select_checkpoint.call(directory.clone())
                    },
                    if selected { "Selected" } else { "Select" }
                }
            }
        }
    }
}

#[component]
fn GameView(
    human_game: GameState,
    ai_game: GameState,
    has_brain: bool,
    stats: SwarmStats,
    player_best_score: u32,
    ai_best_score: u32,
    current_dir: String,
    policy_path: String,
    archive_dir: String,
    operator_log: String,
    alerts: Vec<LiveAlert>,
    on_reset_player: EventHandler<MouseEvent>,
    on_save: EventHandler<MouseEvent>,
    on_reset_trainer: EventHandler<MouseEvent>,
) -> Element {
    rsx! {
        div {
            class: "game-layout",

            main {
                class: "panel game-main",
                div {
                    class: "section-heading",
                    h2 { "Arena" }
                    p { "Manual play on the left. Current best policy on the right." }
                }

                div {
                    class: "versus-grid",

                    div {
                        class: "arena-panel",
                        div {
                            class: "arena-panel__meta",
                            div {
                                h3 { class: "arena-panel__title", "Operator Board" }
                                p { class: "arena-panel__subtitle", "Manual control path" }
                            }
                            div {
                                class: "arena-panel__stats",
                                span { "Score {human_game.score}" }
                                span { "Best {player_best_score}" }
                                span { "Length {human_game.snake.len()}" }
                            }
                        }
                        Board {
                            state: human_game,
                            cell_size: PLAYER_CELL_SIZE,
                            variant: "board-canvas--player".to_string(),
                            snake_color: "#7fd0d8".to_string(),
                            head_color: "#d8f8ff".to_string(),
                            food_color: "#ffcf5a".to_string(),
                        }
                    }

                    div {
                        class: "arena-panel",
                        div {
                            class: "arena-panel__meta",
                            div {
                                h3 { class: "arena-panel__title", "Current Best Policy" }
                                p {
                                    class: "arena-panel__subtitle",
                                    {
                                        if has_brain {
                                            format!("Background swarm champion · {}", stats.arena_stage)
                                        } else {
                                            "Trainer seeding initial policy".to_string()
                                        }
                                    }
                                }
                            }
                            div {
                                class: "arena-panel__stats",
                                span { "Gen {stats.generation}" }
                                span { "Best {ai_best_score}" }
                                span { "Champ {stats.champion_score}" }
                            }
                        }
                        Board {
                            state: ai_game,
                            cell_size: OPPONENT_CELL_SIZE,
                            variant: "board-canvas--ai".to_string(),
                            snake_color: "#a9c58b".to_string(),
                            head_color: "#ebffd0".to_string(),
                            food_color: "#f8b66b".to_string(),
                        }
                    }
                }
            }

            aside {
                class: "panel game-sidebar",

                div {
                    class: "section-heading",
                    h2 { "Controls" }
                    p { "Manual reset for you, autonomous reset for the trainer." }
                }

                div {
                    class: "button-stack",
                    button {
                        class: "control-button",
                        onclick: move |event| on_reset_player.call(event),
                        "Reset Player Board"
                    }
                    button {
                        class: "control-button",
                        onclick: move |event| on_save.call(event),
                        "Save Current Checkpoint"
                    }
                    button {
                        class: "control-button control-button--warn",
                        onclick: move |event| on_reset_trainer.call(event),
                        "Reset Trainer"
                    }
                }

                div {
                    class: "subpanel",
                    h3 { "Live Metrics" }
                    MetricCard {
                        label: "Champion Fitness".to_string(),
                        value: format!("{:.2}", stats.champion_fitness),
                        detail: "Current best model fitness".to_string(),
                    }
                    MetricCard {
                        label: "Mean Fitness".to_string(),
                        value: format!("{:.2}", stats.mean_fitness),
                        detail: "Population average".to_string(),
                    }
                    MetricCard {
                        label: "Steps / Second".to_string(),
                        value: format!("{:.0}", stats.steps_per_second),
                        detail: "Background trainer throughput".to_string(),
                    }
                }

                div {
                    class: "subpanel",
                    h3 { "Checkpoint Paths" }
                    PathItem { label: "Current Dir".to_string(), value: current_dir }
                    PathItem { label: "Policy File".to_string(), value: policy_path }
                    PathItem { label: "Archive Dir".to_string(), value: archive_dir }
                }

                div {
                    class: "subpanel",
                    h3 { "Session" }
                    p { class: "subpanel__copy", "{operator_log}" }
                }

                AlertFeed {
                    title: "Live Alerts".to_string(),
                    alerts,
                }
            }
        }
    }
}

#[component]
fn LabView(
    stats: SwarmStats,
    snapshot: SwarmSnapshot,
    checkpoints: Vec<CheckpointInfo>,
    current_checkpoint: Option<CheckpointInfo>,
    selected_checkpoint: Option<CheckpointInfo>,
    compare_report: Option<CompareReport>,
    replay: Option<LoadedReplay>,
    replay_frame: Option<GameState>,
    replay_index: usize,
    replay_playing: bool,
    lab_best_score: u32,
    artifacts_root: String,
    current_dir: String,
    policy_path: String,
    archive_dir: String,
    alerts: Vec<LiveAlert>,
    on_select_checkpoint: EventHandler<String>,
    on_load_checkpoint: EventHandler<MouseEvent>,
    on_compare_checkpoint: EventHandler<MouseEvent>,
    on_load_replay: EventHandler<MouseEvent>,
    on_replay_prev: EventHandler<MouseEvent>,
    on_replay_next: EventHandler<MouseEvent>,
    on_toggle_replay: EventHandler<MouseEvent>,
) -> Element {
    let max_heat = snapshot.head_heatmap.iter().copied().max().unwrap_or(0);
    let matrix_columns = crate::swarm::POPULATION_MATRIX_COLUMNS;

    rsx! {
        div {
            class: "lab-layout",

            aside {
                class: "panel lab-sidebar",

                div {
                    class: "section-heading",
                    h2 { "Telemetry" }
                    p { "Population-wide counters from the trainer loop." }
                }

                div {
                    class: "metric-row metric-row--stacked",
                    MetricCard {
                        label: "Generation".to_string(),
                        value: stats.generation.to_string(),
                        detail: "Completed breeding cycles".to_string(),
                    }
                    MetricCard {
                        label: "Population".to_string(),
                        value: stats.population_size.to_string(),
                        detail: "Concurrent snake agents".to_string(),
                    }
                    MetricCard {
                        label: "Alive Agents".to_string(),
                        value: stats.alive_agents.to_string(),
                        detail: "Agents still alive in latest snapshot".to_string(),
                    }
                    MetricCard {
                        label: "Curriculum Stage".to_string(),
                        value: stats.arena_stage.clone(),
                        detail: "Current training arena".to_string(),
                    }
                    MetricCard {
                        label: "Best Score".to_string(),
                        value: lab_best_score.to_string(),
                        detail: "Highest archived champion score".to_string(),
                    }
                    MetricCard {
                        label: "Total Steps".to_string(),
                        value: stats.total_agent_steps.to_string(),
                        detail: "Cumulative environment steps".to_string(),
                    }
                }

                div {
                    class: "subpanel",
                    h3 { "Storage" }
                    PathItem { label: "Artifacts Root".to_string(), value: artifacts_root }
                    PathItem { label: "Current Dir".to_string(), value: current_dir }
                    PathItem { label: "Policy File".to_string(), value: policy_path }
                    PathItem { label: "Archive Dir".to_string(), value: archive_dir }
                }
            }

            main {
                class: "panel lab-main",

                div {
                    class: "section-heading",
                    h2 { "Population Matrix" }
                    p { "Every square below represents one background snake. Bright cells indicate higher-scoring live candidates." }
                }

                div {
                    class: "population-matrix",
                    style: "grid-template-columns: repeat({matrix_columns}, minmax(0, 1fr));",
                    for cell in snapshot.population_cells.iter() {
                        div {
                            class: "population-cell {population_cell_modifier(cell)}",
                            title: "agent {cell.id} | score {cell.score} | foods {cell.foods} | fitness {cell.fitness}",
                        }
                    }
                }

                div {
                    class: "section-heading section-heading--spaced",
                    h2 { "Sampled Boards" }
                    p { "A subset of the running population for qualitative inspection." }
                }

                div {
                    class: "sample-grid",
                    for sample in snapshot.sample_agents.iter().take(30) {
                        div {
                            class: "sample-board",
                            div {
                                class: "sample-board__meta",
                                span { "#{sample.id}" }
                                span { "s {sample.state.score}" }
                            }
                            Board {
                                state: sample.state.clone(),
                                cell_size: SAMPLE_CELL_SIZE,
                                variant: "board-canvas--sample".to_string(),
                                snake_color: "#8ea87a".to_string(),
                                head_color: "#d8efbd".to_string(),
                                food_color: "#f0c56c".to_string(),
                            }
                        }
                    }
                }
            }

            aside {
                class: "panel lab-sidebar",

                div {
                    class: "section-heading",
                    h2 { "Champion" }
                    p { "Best policy snapshot currently held in memory." }
                }

                if let Some(champion) = &snapshot.champion {
                    div {
                        class: "champion-card",
                        div {
                            class: "champion-card__meta",
                            span { {format!("Fitness {:.2}", champion.fitness)} }
                            span { "Score {champion.state.score}" }
                        }
                        Board {
                            state: champion.state.clone(),
                            cell_size: CHAMPION_CELL_SIZE,
                            variant: "board-canvas--champion".to_string(),
                            snake_color: "#cfbd89".to_string(),
                            head_color: "#fff0b5".to_string(),
                            food_color: "#ff955c".to_string(),
                        }
                    }
                } else {
                    div {
                        class: "empty-state",
                        "Champion snapshot not available yet."
                    }
                }

                div {
                    class: "subpanel",
                    h3 { "Head Density Heatmap" }
                    div {
                        class: "heatmap-grid",
                        for cell in snapshot.head_heatmap.iter() {
                            div {
                                class: "heatmap-cell",
                                style: "background: {heatmap_color(*cell, max_heat)};",
                            }
                        }
                    }
                }

                div {
                    class: "subpanel",
                    h3 { "Checkpoint Manager" }
                    p {
                        class: "subpanel__copy",
                        "Promote archived models into the active slot or compare them over a fixed evaluation seed pack."
                    }

                    div {
                        class: "checkpoint-meta-grid",
                        CheckpointMetaCard {
                            title: "Current Slot".to_string(),
                            checkpoint: current_checkpoint.clone(),
                        }
                        CheckpointMetaCard {
                            title: "Selected Archive".to_string(),
                            checkpoint: selected_checkpoint.clone(),
                        }
                    }

                    div {
                        class: "button-stack",
                        button {
                            class: "control-button",
                            disabled: selected_checkpoint.is_none(),
                            onclick: move |event| on_load_checkpoint.call(event),
                            "Load Selected Checkpoint"
                        }
                        button {
                            class: "control-button",
                            disabled: selected_checkpoint.is_none(),
                            onclick: move |event| on_compare_checkpoint.call(event),
                            "Compare Against Current"
                        }
                        button {
                            class: "control-button",
                            disabled: selected_checkpoint.is_none(),
                            onclick: move |event| on_load_replay.call(event),
                            "Load Replay"
                        }
                    }

                    if let Some(report) = compare_report.clone() {
                        CompareSummaryCard { report }
                    }
                }

                div {
                    class: "subpanel",
                    h3 { "Replay Viewer" }
                    if let Some(replay) = replay.clone() {
                        p {
                            class: "subpanel__copy",
                            "Generation {replay.source.generation} replay over seed {replay.trace.seed}. Frame {replay_index + 1} of {replay.frames.len()}."
                        }
                        div {
                            class: "button-stack replay-controls",
                            button {
                                class: "control-button",
                                onclick: move |event| on_replay_prev.call(event),
                                "Prev"
                            }
                            button {
                                class: "control-button",
                                onclick: move |event| on_toggle_replay.call(event),
                                if replay_playing { "Pause" } else { "Play" }
                            }
                            button {
                                class: "control-button",
                                onclick: move |event| on_replay_next.call(event),
                                "Next"
                            }
                        }

                        if let Some(frame) = replay_frame.clone() {
                            Board {
                                state: frame,
                                cell_size: REPLAY_CELL_SIZE,
                                variant: "board-canvas--champion".to_string(),
                                snake_color: "#96a27e".to_string(),
                                head_color: "#edf0cf".to_string(),
                                food_color: "#efad68".to_string(),
                            }
                        }
                    } else {
                        div {
                            class: "empty-state",
                            "Select a checkpoint and load its replay."
                        }
                    }
                }

                div {
                    class: "archive-panel",
                    div {
                        class: "section-heading",
                        h2 { "Recent Checkpoints" }
                        p { "Most recent archived generations written to disk." }
                    }
                    div {
                        class: "archive-grid",
                        for checkpoint in checkpoints {
                            ArchiveCheckpointEntry {
                                checkpoint: checkpoint.clone(),
                                selected: selected_checkpoint
                                    .as_ref()
                                    .map(|selected| selected.directory == checkpoint.directory)
                                    .unwrap_or(false),
                                on_select_checkpoint: move |directory| on_select_checkpoint.call(directory),
                            }
                        }
                    }
                }

                AlertFeed {
                    title: "Live Alerts".to_string(),
                    alerts,
                }
            }
        }
    }
}

#[component]
fn AlertFeed(title: String, alerts: Vec<LiveAlert>) -> Element {
    rsx! {
        div {
            class: "subpanel",
            h3 { "{title}" }
            if alerts.is_empty() {
                div {
                    class: "empty-state",
                    "No alerts yet."
                }
            } else {
                div {
                    class: "alert-feed",
                    for alert in alerts {
                        div {
                            key: "{alert.id}",
                            class: "{alert_level_class(alert.level)}",
                            div {
                                class: "alert-card__top",
                                span { class: "alert-card__tag", "{alert_level_label(alert.level)}" }
                                span { class: "alert-card__headline", "{alert.headline}" }
                            }
                            p { class: "alert-card__detail", "{alert.detail}" }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn Board(
    state: GameState,
    cell_size: i32,
    variant: String,
    snake_color: String,
    head_color: String,
    food_color: String,
) -> Element {
    let board_size = state.grid_size * cell_size;
    let food_radius = (cell_size / 2).max(2);

    rsx! {
        div {
            class: "board-canvas {variant}",
            style: "width: {board_size}px; height: {board_size}px;",

            for obstacle in state.arena.obstacles.iter() {
                div {
                    class: "board-obstacle",
                    style: "width: {cell_size}px; height: {cell_size}px; left: {obstacle.x * cell_size}px; top: {obstacle.y * cell_size}px;",
                }
            }

            div {
                class: "board-food",
                style: "width: {cell_size}px; height: {cell_size}px; left: {state.food.x * cell_size}px; top: {state.food.y * cell_size}px; background: {food_color}; border-radius: {food_radius}px;",
            }

            for (index, segment) in state.snake.iter().enumerate() {
                div {
                    class: "board-segment",
                    style: {
                        let segment_color = if index == 0 {
                            head_color.as_str()
                        } else {
                            snake_color.as_str()
                        };
                        segment_style(segment, cell_size, segment_color, 1.0 - (index as f32 * 0.035).min(0.5))
                    },
                }
            }

            if state.is_dead {
                div {
                    class: "board-overlay",
                    "Dead"
                }
            }
        }
    }
}
