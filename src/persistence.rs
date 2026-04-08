use crate::ai::PolicyBrain;
use crate::swarm::SwarmStats;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const ARTIFACTS_ROOT: &str = "artifacts/checkpoints";
const CURRENT_DIR: &str = "artifacts/checkpoints/current";
const ARCHIVE_DIR: &str = "artifacts/checkpoints/archive";
const POLICY_FILE: &str = "policy.bin";
const STATS_FILE: &str = "stats.bin";
const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersistentData {
    pub brain: Option<PolicyBrain>,
    pub stats: SwarmStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckpointManifest {
    pub generation: u32,
    pub population_size: u32,
    pub alive_agents: u32,
    pub champion_score: u32,
    pub champion_foods: u32,
    pub champion_fitness: f32,
    pub mean_fitness: f32,
    pub median_fitness: f32,
    pub mutation_scale: f32,
    pub total_agent_steps: u64,
    pub total_foods: u64,
    pub parameter_count: usize,
    pub saved_at: String,
    pub save_kind: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointInfo {
    pub generation: u32,
    pub champion_score: u32,
    pub champion_foods: u32,
    pub champion_fitness: f32,
    pub mean_fitness: f32,
    pub median_fitness: f32,
    pub mutation_scale: f32,
    pub saved_at: String,
    pub save_kind: String,
    pub directory: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SaveReport {
    pub current_dir: String,
    pub policy_path: String,
    pub has_brain: bool,
}

pub fn artifacts_root_display() -> String {
    PathBuf::from(ARTIFACTS_ROOT).display().to_string()
}

pub fn current_checkpoint_dir_display() -> String {
    PathBuf::from(CURRENT_DIR).display().to_string()
}

pub fn current_policy_path_display() -> String {
    PathBuf::from(CURRENT_DIR)
        .join(POLICY_FILE)
        .display()
        .to_string()
}

pub fn archive_dir_display() -> String {
    PathBuf::from(ARCHIVE_DIR).display().to_string()
}

pub fn save_current(
    brain: &Option<PolicyBrain>,
    stats: &SwarmStats,
) -> Result<SaveReport, Box<dyn std::error::Error>> {
    let current_dir = PathBuf::from(CURRENT_DIR);
    save_bundle(&current_dir, brain, stats, "manual-save")?;

    Ok(SaveReport {
        current_dir: current_dir.display().to_string(),
        policy_path: current_dir.join(POLICY_FILE).display().to_string(),
        has_brain: brain.is_some(),
    })
}

pub fn save_generation_checkpoint(
    brain: &PolicyBrain,
    stats: &SwarmStats,
) -> Result<Option<CheckpointInfo>, Box<dyn std::error::Error>> {
    let checkpoint_dir = PathBuf::from(ARCHIVE_DIR).join(format!(
        "gen-{generation:05}-score-{score:04}-{stamp}",
        generation = stats.generation,
        score = stats.champion_score,
        stamp = timestamp_slug()
    ));

    let brain = Some(brain.clone());
    save_bundle(&checkpoint_dir, &brain, stats, "generation-checkpoint")?;
    save_bundle(
        &PathBuf::from(CURRENT_DIR),
        &brain,
        stats,
        "autosave-current",
    )?;

    Ok(read_checkpoint(&checkpoint_dir).ok())
}

pub fn load_current() -> PersistentData {
    ensure_dirs().ok();

    let stats =
        read_bin::<SwarmStats>(&PathBuf::from(CURRENT_DIR).join(STATS_FILE)).unwrap_or_default();
    let brain = read_bin::<PolicyBrain>(&PathBuf::from(CURRENT_DIR).join(POLICY_FILE));

    PersistentData { brain, stats }
}

pub fn list_checkpoints(limit: usize) -> Vec<CheckpointInfo> {
    let Ok(entries) = fs::read_dir(ARCHIVE_DIR) else {
        return Vec::new();
    };

    let mut checkpoints: Vec<_> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter_map(|path| read_checkpoint(&path).ok())
        .collect();

    checkpoints.sort_by(|left, right| {
        right
            .generation
            .cmp(&left.generation)
            .then_with(|| right.saved_at.cmp(&left.saved_at))
    });
    checkpoints.truncate(limit);
    checkpoints
}

fn save_bundle(
    dir: &Path,
    brain: &Option<PolicyBrain>,
    stats: &SwarmStats,
    save_kind: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_dirs()?;
    fs::create_dir_all(dir)?;

    write_bin(&dir.join(STATS_FILE), stats)?;
    let policy_path = dir.join(POLICY_FILE);
    if let Some(brain) = brain {
        write_bin(&policy_path, brain)?;
    } else if policy_path.exists() {
        fs::remove_file(policy_path)?;
    }

    let manifest = CheckpointManifest {
        generation: stats.generation,
        population_size: stats.population_size,
        alive_agents: stats.alive_agents,
        champion_score: stats.champion_score,
        champion_foods: stats.champion_foods,
        champion_fitness: stats.champion_fitness,
        mean_fitness: stats.mean_fitness,
        median_fitness: stats.median_fitness,
        mutation_scale: stats.mutation_scale,
        total_agent_steps: stats.total_agent_steps,
        total_foods: stats.total_foods,
        parameter_count: brain
            .as_ref()
            .map(|_| crate::ai::PolicyBrain::parameter_count())
            .unwrap_or(0),
        saved_at: timestamp_display(),
        save_kind: save_kind.to_string(),
    };

    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let mut manifest_file = File::create(dir.join(MANIFEST_FILE))?;
    manifest_file.write_all(&manifest_bytes)?;
    Ok(())
}

fn read_checkpoint(path: &Path) -> Result<CheckpointInfo, Box<dyn std::error::Error>> {
    let manifest: CheckpointManifest = read_json(&path.join(MANIFEST_FILE))?;
    Ok(CheckpointInfo {
        generation: manifest.generation,
        champion_score: manifest.champion_score,
        champion_foods: manifest.champion_foods,
        champion_fitness: manifest.champion_fitness,
        mean_fitness: manifest.mean_fitness,
        median_fitness: manifest.median_fitness,
        mutation_scale: manifest.mutation_scale,
        saved_at: manifest.saved_at,
        save_kind: manifest.save_kind,
        directory: path.display().to_string(),
    })
}

fn ensure_dirs() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(CURRENT_DIR)?;
    fs::create_dir_all(ARCHIVE_DIR)?;
    Ok(())
}

fn write_bin<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = bincode::serialize(value)?;
    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    Ok(())
}

fn read_bin<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let mut file = File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).ok()?;
    bincode::deserialize(&bytes).ok()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn timestamp_display() -> String {
    Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

fn timestamp_slug() -> String {
    Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}
