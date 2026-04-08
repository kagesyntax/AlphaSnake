mod ai;
mod game;
mod persistence;
mod swarm;

use crate::ai::{would_collide, PolicyBrain};
use crate::game::{Direction, GameState, Pos};
use crate::persistence::{
    archive_dir_display, artifacts_root_display, current_checkpoint_dir_display,
    current_policy_path_display, list_checkpoints, load_current, save_current, CheckpointInfo,
};
use crate::swarm::{run_swarm, PopulationCell, SwarmSnapshot, SwarmStats};
use dioxus::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

const GRID_SIZE: i32 = 20;
const PLAYER_CELL_SIZE: i32 = 20;
const OPPONENT_CELL_SIZE: i32 = 18;
const CHAMPION_CELL_SIZE: i32 = 12;
const SAMPLE_CELL_SIZE: i32 = 6;
const RESEARCH_CSS: Asset = asset!("/assets/research.css");

fn main() {
    dioxus::launch(App);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Game,
    Lab,
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
    let mut operator_log = use_signal(|| "Research console online".to_string());
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
                tokio::time::sleep(Duration::from_millis(120)).await;

                human_game.with_mut(|game| {
                    if game.is_dead {
                        return;
                    }
                    human_rng.with_mut(|rng| {
                        let _ = game.step(human_dir(), rng);
                    });
                });

                let current_brain = brain_for_game.read().unwrap().clone();
                ai_game.with_mut(move |game| {
                    ai_rng.with_mut(|rng| {
                        advance_ai_game(game, rng, current_brain.as_ref());
                    });
                });
            }
        }
    });

    use_future(move || async move {
        loop {
            tokio::time::sleep(Duration::from_millis(250)).await;
            refresh.with_mut(|tick| *tick += 1);
        }
    });

    let _ = refresh();
    let current_stats = shared_stats.read().unwrap().clone();
    let current_snapshot = shared_snapshot.read().unwrap().clone();
    let has_brain = shared_brain.read().unwrap().is_some();
    let checkpoints = list_checkpoints(12);
    let artifacts_root = absolute_path(&artifacts_root_display());
    let current_dir = absolute_path(&current_checkpoint_dir_display());
    let policy_path = absolute_path(&current_policy_path_display());
    let archive_dir = absolute_path(&archive_dir_display());

    rsx! {
        document::Stylesheet { href: RESEARCH_CSS }
        document::Title { "Snake Systems Lab" }

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
                    current_dir: current_dir.clone(),
                    policy_path: policy_path.clone(),
                    archive_dir: archive_dir.clone(),
                    operator_log: operator_log(),
                    on_reset_player: move |_| {
                        human_game.with_mut(|game| {
                            human_rng.with_mut(|rng| {
                                *game = GameState::new(GRID_SIZE, rng);
                            });
                        });
                        operator_log.set("Player board reset".to_string());
                    },
                    on_save: move |_| {
                        let brain = shared_brain.read().unwrap().clone();
                        let stats = shared_stats.read().unwrap().clone();
                        match save_current(&brain, &stats) {
                            Ok(report) => {
                                if report.has_brain {
                                    operator_log.set(format!("Saved active policy to {}", absolute_path(&report.policy_path)));
                                } else {
                                    operator_log.set(format!("Saved stats only to {}", absolute_path(&report.current_dir)));
                                }
                            }
                            Err(error) => operator_log.set(format!("Save failed: {error}")),
                        }
                    },
                    on_reset_trainer: move |_| {
                        shared_reset_for_button.fetch_add(1, Ordering::Relaxed);
                        operator_log.set("Trainer reset requested".to_string());
                    },
                }
            } else {
                LabView {
                    stats: current_stats.clone(),
                    snapshot: current_snapshot.clone(),
                    checkpoints: checkpoints.clone(),
                    artifacts_root,
                    current_dir,
                    policy_path,
                    archive_dir,
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
fn GameView(
    human_game: GameState,
    ai_game: GameState,
    has_brain: bool,
    stats: SwarmStats,
    current_dir: String,
    policy_path: String,
    archive_dir: String,
    operator_log: String,
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
                                p { class: "arena-panel__subtitle",
                                    if has_brain { "Background swarm champion" } else { "Trainer seeding initial policy" }
                                }
                            }
                            div {
                                class: "arena-panel__stats",
                                span { "Gen {stats.generation}" }
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
            }
        }
    }
}

#[component]
fn LabView(
    stats: SwarmStats,
    snapshot: SwarmSnapshot,
    checkpoints: Vec<CheckpointInfo>,
    artifacts_root: String,
    current_dir: String,
    policy_path: String,
    archive_dir: String,
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
                    class: "archive-panel",
                    div {
                        class: "section-heading",
                        h2 { "Recent Checkpoints" }
                        p { "Most recent archived generations written to disk." }
                    }
                    div {
                        class: "archive-grid",
                        for checkpoint in checkpoints {
                            div {
                                class: "archive-entry",
                                div {
                                    class: "archive-entry__top",
                                    span { "Gen {checkpoint.generation}" }
                                    span { "Score {checkpoint.champion_score}" }
                                }
                                p { class: "archive-entry__line", {format!("Fitness {:.2}", checkpoint.champion_fitness)} }
                                p { class: "archive-entry__line", "{checkpoint.saved_at}" }
                                p { class: "archive-entry__path", "{absolute_path(&checkpoint.directory)}" }
                            }
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
