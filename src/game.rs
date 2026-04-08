use rand::rngs::SmallRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub fn opposite(&self) -> Self {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArenaKind {
    Standard,
    SparseFood,
    ObstacleField,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArenaConfig {
    pub kind: ArenaKind,
    pub grid_size: i32,
    pub min_food_distance: i32,
    pub obstacles: Vec<Pos>,
}

impl ArenaConfig {
    pub fn standard(grid_size: i32) -> Self {
        Self {
            kind: ArenaKind::Standard,
            grid_size,
            min_food_distance: 0,
            obstacles: Vec::new(),
        }
    }

    pub fn sparse_food(grid_size: i32) -> Self {
        Self {
            kind: ArenaKind::SparseFood,
            grid_size,
            min_food_distance: (grid_size / 2).max(6),
            obstacles: Vec::new(),
        }
    }

    pub fn obstacle_field(grid_size: i32) -> Self {
        let center = grid_size / 2;
        let obstacles = vec![
            Pos {
                x: center - 3,
                y: center - 2,
            },
            Pos {
                x: center - 3,
                y: center - 1,
            },
            Pos {
                x: center - 3,
                y: center,
            },
            Pos {
                x: center + 2,
                y: center,
            },
            Pos {
                x: center + 2,
                y: center + 1,
            },
            Pos {
                x: center + 2,
                y: center + 2,
            },
        ];

        Self {
            kind: ArenaKind::ObstacleField,
            grid_size,
            min_food_distance: 0,
            obstacles,
        }
    }

    pub fn curriculum_stage(stage: usize, grid_size: i32) -> Self {
        match stage {
            1 => Self::sparse_food(grid_size),
            2 => Self::obstacle_field(grid_size),
            _ => Self::standard(grid_size),
        }
    }

    pub fn stage_label(&self) -> &'static str {
        match self.kind {
            ArenaKind::Standard => "Standard Arena",
            ArenaKind::SparseFood => "Sparse Food Arena",
            ArenaKind::ObstacleField => "Obstacle Field",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GameState {
    pub snake: VecDeque<Pos>,
    pub food: Pos,
    pub direction: Direction,
    pub grid_size: i32,
    pub is_dead: bool,
    pub score: u32,
    pub foods_eaten: u32,
    pub steps_alive: u32,
    pub steps_since_food: u32,
    pub arena: ArenaConfig,
}

impl GameState {
    pub fn new(grid_size: i32, rng: &mut SmallRng) -> Self {
        Self::new_with_arena(ArenaConfig::standard(grid_size), rng)
    }

    pub fn new_with_arena(arena: ArenaConfig, rng: &mut SmallRng) -> Self {
        let mut snake = VecDeque::new();
        let grid_size = arena.grid_size;
        let mid = grid_size / 2;
        snake.push_back(Pos { x: mid, y: mid });
        snake.push_back(Pos { x: mid, y: mid + 1 });

        let mut state = Self {
            snake,
            food: Pos { x: 0, y: 0 },
            direction: Direction::Up,
            grid_size,
            is_dead: false,
            score: 0,
            foods_eaten: 0,
            steps_alive: 0,
            steps_since_food: 0,
            arena,
        };
        state.spawn_food(rng);
        state
    }

    pub fn head(&self) -> Pos {
        *self.snake.front().expect("snake must have a head")
    }

    pub fn starvation_limit(&self) -> u32 {
        (self.grid_size * self.grid_size * 2) as u32
    }

    pub fn would_collide(&self, next_head: Pos) -> bool {
        if next_head.x < 0
            || next_head.x >= self.grid_size
            || next_head.y < 0
            || next_head.y >= self.grid_size
        {
            return true;
        }

        if self.arena.obstacles.contains(&next_head) {
            return true;
        }

        let tail = self.snake.back().copied();
        self.snake.iter().enumerate().any(|(index, segment)| {
            if *segment != next_head {
                return false;
            }
            let moving_into_tail = Some(next_head) == tail && index == self.snake.len() - 1;
            !moving_into_tail
        })
    }

    pub fn distance_to_food(&self) -> i32 {
        let head = self.head();
        (head.x - self.food.x).abs() + (head.y - self.food.y).abs()
    }

    pub fn step(&mut self, next_dir: Direction, rng: &mut SmallRng) -> StepOutcome {
        if self.is_dead {
            return StepOutcome::Dead;
        }

        let previous_distance = self.distance_to_food();
        if next_dir != self.direction.opposite() {
            self.direction = next_dir;
        }

        let head = self.head();
        let new_head = match self.direction {
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
        };

        self.steps_alive += 1;
        self.steps_since_food += 1;

        if self.would_collide(new_head) {
            self.is_dead = true;
            return StepOutcome::Dead;
        }

        self.snake.push_front(new_head);
        let ate_food = new_head == self.food;

        if ate_food {
            self.score += 10;
            self.foods_eaten += 1;
            self.steps_since_food = 0;
            self.spawn_food(rng);
        } else {
            self.snake.pop_back();
        }

        if self.steps_since_food > self.starvation_limit() {
            self.is_dead = true;
            return StepOutcome::Dead;
        }

        let distance_delta = previous_distance - self.distance_to_food();
        if ate_food {
            StepOutcome::AteFood
        } else if distance_delta > 0 {
            StepOutcome::Closer(distance_delta)
        } else if distance_delta < 0 {
            StepOutcome::Further(distance_delta.abs())
        } else {
            StepOutcome::Neutral
        }
    }

    pub fn spawn_food(&mut self, rng: &mut SmallRng) {
        let head = self.head();
        for _ in 0..256 {
            let new_food = Pos {
                x: rng.gen_range(0..self.grid_size),
                y: rng.gen_range(0..self.grid_size),
            };
            let distance = (head.x - new_food.x).abs() + (head.y - new_food.y).abs();
            if !self.snake.contains(&new_food)
                && !self.arena.obstacles.contains(&new_food)
                && distance >= self.arena.min_food_distance
            {
                self.food = new_food;
                break;
            }
        }

        if self.snake.contains(&self.food) || self.arena.obstacles.contains(&self.food) {
            loop {
                let new_food = Pos {
                    x: rng.gen_range(0..self.grid_size),
                    y: rng.gen_range(0..self.grid_size),
                };
                if !self.snake.contains(&new_food) && !self.arena.obstacles.contains(&new_food) {
                    self.food = new_food;
                    break;
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    AteFood,
    Closer(i32),
    Further(i32),
    Neutral,
    Dead,
}
