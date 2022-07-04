use rand::Rng;
use std::f64::consts;
use tch::Tensor;

pub const EXPLORATION_START_PROB: f64 = 0.9;
pub const EXPLORATION_END_PROB: f64 = 0.05;

pub const GAMMA: f64 = 0.999;

pub trait Agent {
    fn select_action(&self, obs: &Tensor) -> i64;
    fn consume_event(
        &mut self,
        obs: &Tensor,
        next_obs: Option<&Tensor>,
        reward: f64,
        action: i64,
        is_done: bool,
    ) -> f64; // returns loss on episode end
    fn forward(&self, obs: &Tensor) -> Tensor;
}
pub fn need_eps_greedy_exploration(steps: i64, exploration_prob_decay_steps: i64) -> bool {
    let mut rng = rand::thread_rng();
    let rand: f64 = rng.gen_range(0.0..1.0);
    let explore_prob = EXPLORATION_END_PROB
        + (EXPLORATION_START_PROB - EXPLORATION_END_PROB)
            * (consts::E).powf(-1.0f64 * steps as f64 / exploration_prob_decay_steps as f64);
    if rand <= explore_prob {
        true
    } else {
        false
    }
}

pub struct Event {
    pub observation: Tensor,
    pub reward: f64,
    pub action: i64,
    pub is_done: bool,
}

pub struct MemoryEvent {
    pub observation: Tensor,
    pub value: f64,
    pub action: i64,
}

pub fn calc_values(events: &Vec<Event>) -> Vec<f64> {
    let mut cumreward: f64 = 0.0;
    let mut values: Vec<f64> = vec![0.0; events.len()];
    for i in (0..events.len()).rev() {
        if events[i].is_done {
            cumreward = 0.0;
        }
        values[i] = events[i].reward + cumreward * GAMMA;
        cumreward = cumreward * GAMMA + events[i].reward;
    }
    values
}
