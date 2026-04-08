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
const REGISTRY_FILE: &str = "artifacts/model-registry.json";
const POLICY_FILE: &str = "policy.bin";
const STATS_FILE: &str = "stats.bin";
const MANIFEST_FILE: &str = "manifest.json";
const REPLAY_FILE: &str = "replay.json";

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
    #[serde(default = "default_model_id")]
    pub model_id: String,
    #[serde(default)]
    pub parent_model_id: Option<String>,
    #[serde(default = "default_registry_status")]
    pub registry_status: String,
    #[serde(default = "default_arena_stage")]
    pub arena_stage: String,
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
    pub model_id: String,
    pub parent_model_id: Option<String>,
    pub registry_status: String,
    pub arena_stage: String,
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
pub struct CheckpointBundle {
    pub info: CheckpointInfo,
    pub brain: Option<PolicyBrain>,
    pub stats: SwarmStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SaveReport {
    pub current_dir: String,
    pub policy_path: String,
    pub has_brain: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromoteReport {
    pub save: SaveReport,
    pub data: PersistentData,
    pub source: CheckpointInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelRecord {
    pub model_id: String,
    pub parent_model_id: Option<String>,
    pub label: String,
    pub status: String,
    pub generation: u32,
    pub champion_score: u32,
    pub champion_fitness: f32,
    pub arena_stage: String,
    pub saved_at: String,
    pub checkpoint_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ModelRegistry {
    pub models: Vec<ModelRecord>,
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

pub fn list_registry(limit: usize) -> Vec<ModelRecord> {
    let mut registry = read_registry();
    registry.models.sort_by(|left, right| {
        right
            .generation
            .cmp(&left.generation)
            .then_with(|| right.saved_at.cmp(&left.saved_at))
    });
    registry.models.truncate(limit);
    registry.models
}

pub fn save_current(
    brain: &Option<PolicyBrain>,
    stats: &SwarmStats,
) -> Result<SaveReport, Box<dyn std::error::Error>> {
    let current_dir = PathBuf::from(CURRENT_DIR);
    let existing = current_checkpoint_info();
    let model_id = existing
        .as_ref()
        .map(|info| info.model_id.clone())
        .unwrap_or_else(|| new_model_id(stats.generation));
    let parent_model_id = existing.as_ref().and_then(|info| info.parent_model_id.clone());
    let registry_status = existing
        .as_ref()
        .map(|info| info.registry_status.clone())
        .unwrap_or_else(|| "candidate".to_string());

    save_bundle(
        &current_dir,
        brain,
        stats,
        "manual-save",
        &model_id,
        parent_model_id.as_deref(),
        &registry_status,
    )?;
    if let Some(brain) = brain {
        save_replay_bundle(&current_dir, &crate::replay::record_replay(brain))?;
        upsert_registry(ModelRecord {
            model_id,
            parent_model_id,
            label: format!("Manual Save G{}", stats.generation),
            status: registry_status,
            generation: stats.generation,
            champion_score: stats.champion_score,
            champion_fitness: stats.champion_fitness,
            arena_stage: stats.arena_stage.clone(),
            saved_at: timestamp_display(),
            checkpoint_dir: current_dir.display().to_string(),
        })?;
    }

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
    let parent_model_id = current_checkpoint_info()
        .map(|info| info.model_id)
        .filter(|model_id| model_id != "untracked");
    let model_id = new_model_id(stats.generation);
    save_bundle(
        &checkpoint_dir,
        &brain,
        stats,
        "generation-checkpoint",
        &model_id,
        parent_model_id.as_deref(),
        "champion",
    )?;
    save_replay_bundle(&checkpoint_dir, &crate::replay::record_replay(brain.as_ref().unwrap()))?;
    save_bundle(
        &PathBuf::from(CURRENT_DIR),
        &brain,
        stats,
        "autosave-current",
        &model_id,
        parent_model_id.as_deref(),
        "champion",
    )?;
    save_replay_bundle(&PathBuf::from(CURRENT_DIR), &crate::replay::record_replay(brain.as_ref().unwrap()))?;
    promote_registry_model(
        ModelRecord {
            model_id,
            parent_model_id,
            label: format!("Champion G{}", stats.generation),
            status: "champion".to_string(),
            generation: stats.generation,
            champion_score: stats.champion_score,
            champion_fitness: stats.champion_fitness,
            arena_stage: stats.arena_stage.clone(),
            saved_at: timestamp_display(),
            checkpoint_dir: checkpoint_dir.display().to_string(),
        },
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

pub fn current_checkpoint_info() -> Option<CheckpointInfo> {
    read_checkpoint(Path::new(CURRENT_DIR)).ok()
}

pub fn load_checkpoint_bundle(directory: &str) -> Result<CheckpointBundle, Box<dyn std::error::Error>> {
    let path = PathBuf::from(directory);
    let info = read_checkpoint(&path)?;
    let stats = read_bin::<SwarmStats>(&path.join(STATS_FILE))
        .ok_or_else(|| format!("missing stats in {}", path.display()))?;
    let brain = read_bin::<PolicyBrain>(&path.join(POLICY_FILE));

    Ok(CheckpointBundle { info, brain, stats })
}

pub fn promote_checkpoint(directory: &str) -> Result<PromoteReport, Box<dyn std::error::Error>> {
    let bundle = load_checkpoint_bundle(directory)?;
    save_bundle(
        &PathBuf::from(CURRENT_DIR),
        &bundle.brain,
        &bundle.stats,
        "promoted-from-archive",
        &bundle.info.model_id,
        bundle.info.parent_model_id.as_deref(),
        "champion",
    )?;
    if let Some(brain) = &bundle.brain {
        save_replay_bundle(&PathBuf::from(CURRENT_DIR), &crate::replay::record_replay(brain))?;
    }
    promote_registry_model(ModelRecord {
        model_id: bundle.info.model_id.clone(),
        parent_model_id: bundle.info.parent_model_id.clone(),
        label: format!("Promoted G{}", bundle.info.generation),
        status: "champion".to_string(),
        generation: bundle.info.generation,
        champion_score: bundle.info.champion_score,
        champion_fitness: bundle.info.champion_fitness,
        arena_stage: bundle.info.arena_stage.clone(),
        saved_at: timestamp_display(),
        checkpoint_dir: bundle.info.directory.clone(),
    })?;

    let save = SaveReport {
        current_dir: PathBuf::from(CURRENT_DIR).display().to_string(),
        policy_path: PathBuf::from(CURRENT_DIR).join(POLICY_FILE).display().to_string(),
        has_brain: bundle.brain.is_some(),
    };

    Ok(PromoteReport {
        save,
        data: PersistentData {
            brain: bundle.brain.clone(),
            stats: bundle.stats.clone(),
        },
        source: bundle.info,
    })
}

pub fn load_replay(
    directory: &str,
) -> Result<crate::replay::ReplayTrace, Box<dyn std::error::Error>> {
    read_json(&PathBuf::from(directory).join(REPLAY_FILE))
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
    model_id: &str,
    parent_model_id: Option<&str>,
    registry_status: &str,
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
        model_id: model_id.to_string(),
        parent_model_id: parent_model_id.map(str::to_string),
        registry_status: registry_status.to_string(),
        arena_stage: stats.arena_stage.clone(),
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

fn save_replay_bundle(
    dir: &Path,
    replay: &crate::replay::ReplayTrace,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = serde_json::to_vec_pretty(replay)?;
    let mut file = File::create(dir.join(REPLAY_FILE))?;
    file.write_all(&bytes)?;
    Ok(())
}

fn read_checkpoint(path: &Path) -> Result<CheckpointInfo, Box<dyn std::error::Error>> {
    let manifest: CheckpointManifest = read_json(&path.join(MANIFEST_FILE))?;
    Ok(CheckpointInfo {
        generation: manifest.generation,
        model_id: manifest.model_id,
        parent_model_id: manifest.parent_model_id,
        registry_status: manifest.registry_status,
        arena_stage: manifest.arena_stage,
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

fn default_arena_stage() -> String {
    "Standard Arena".to_string()
}

fn default_model_id() -> String {
    "untracked".to_string()
}

fn default_registry_status() -> String {
    "candidate".to_string()
}

fn new_model_id(generation: u32) -> String {
    format!("mdl-{generation:05}-{}", timestamp_slug())
}

fn read_registry() -> ModelRegistry {
    read_json(&PathBuf::from(REGISTRY_FILE)).unwrap_or_default()
}

fn write_registry(registry: &ModelRegistry) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(REGISTRY_FILE).parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(registry)?;
    let mut file = File::create(REGISTRY_FILE)?;
    file.write_all(&bytes)?;
    Ok(())
}

fn upsert_registry(record: ModelRecord) -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = read_registry();
    if let Some(existing) = registry
        .models
        .iter_mut()
        .find(|existing| existing.model_id == record.model_id)
    {
        *existing = record;
    } else {
        registry.models.push(record);
    }
    write_registry(&registry)
}

fn promote_registry_model(record: ModelRecord) -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = read_registry();
    for existing in &mut registry.models {
        if existing.status == "champion" && existing.model_id != record.model_id {
            existing.status = "retired".to_string();
        }
    }

    if let Some(existing) = registry
        .models
        .iter_mut()
        .find(|existing| existing.model_id == record.model_id)
    {
        *existing = record;
    } else {
        registry.models.push(record);
    }

    write_registry(&registry)
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
