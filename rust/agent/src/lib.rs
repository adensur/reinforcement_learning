use rand::Rng;
use std::collections::vec_deque::VecDeque;
use std::f64::consts;
use tch::{kind, nn, nn::ModuleT, nn::OptimizerConfig, Device, Kind, Tensor};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

const INPUT_DIMS: i64 = 4;
const OUTPUT_DIMS: i64 = 2;
const GAMMA: f64 = 0.999;

const EXPLORATION_START_PROB: f64 = 0.9;
const EXPLORATION_END_PROB: f64 = 0.05;
const EXPLORATION_PROB_DECAY_STEPS: f64 = 100_000.0;

const MEMORY_CAPACITY: usize = 1_000;
const BATCH_SIZE: usize = 128;

#[derive(Debug)]
pub struct Network {
    pub input_layer: nn::Linear,
    pub layers: Vec<nn::Linear>,
    pub final_layer: nn::Linear,
}

impl Network {
    fn new(vs: &nn::Path, hidden_size: i64, hidden_layers: usize) -> Network {
        let input_layer = nn::linear(vs, INPUT_DIMS, hidden_size, Default::default());
        let mut layers: Vec<nn::Linear> = Vec::new();
        for _ in 0..hidden_layers {
            layers.push(nn::linear(vs, hidden_size, hidden_size, Default::default()));
        }
        let final_layer = nn::linear(vs, hidden_size, OUTPUT_DIMS, Default::default());
        Network {
            input_layer: input_layer,
            layers: layers,
            final_layer: final_layer,
        }
    }
}

impl nn::ModuleT for Network {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let mut t = xs.apply(&self.input_layer).relu();
        for layer in &self.layers {
            t = t.apply(layer).relu();
        }
        t.apply(&self.final_layer)
    }
}

struct Event {
    observation: Tensor,
    reward: f64,
    action: i64,
}

struct MemoryEvent {
    observation: Tensor,
    value: f64,
    action: i64,
}

pub struct SimpleAgent {
    pub network: Network,
    events_buf: Vec<Event>,
    optimizer: tch::nn::Optimizer,
    steps: i64,
    memory: VecDeque<MemoryEvent>,
}

fn sdprint1(t: &Tensor) -> String {
    if t.dim() != 1 {
        panic!("Expected tensor of rank 1!");
    }
    let mut res = String::new();
    for i in 0..t.size1().unwrap() {
        res.push_str(&format!("{}, ", t.double_value(&[i])));
    }
    res
}

impl SimpleAgent {
    pub fn new() -> Self {
        let vs = nn::VarStore::new(Device::Cpu);
        let opt = nn::RmsProp::default().build(&vs, 1e-2).unwrap();
        SimpleAgent {
            network: Network::new(&vs.root(), 32, 2),
            events_buf: Vec::new(),
            optimizer: opt,
            steps: 0,
            memory: VecDeque::with_capacity(MEMORY_CAPACITY),
        }
    }

    pub fn select_action(&self, obs: &Tensor) -> i64 {
        let mut rng = rand::thread_rng();
        let rand: f64 = rng.gen_range(0.0..1.0);
        let explore_prob = EXPLORATION_END_PROB
            + (EXPLORATION_START_PROB - EXPLORATION_END_PROB)
                * (consts::E).powf(-1.0f64 * self.steps as f64 / EXPLORATION_PROB_DECAY_STEPS);
        if rand <= explore_prob {
            let action = rng.gen_range(0..=1);
            action
        } else {
            let pred = self.network.forward_t(obs, false);
            pred.max_dim(0, false).1.int64_value(&[])
        }
    }

    pub fn consume_event(&mut self, obs: &Tensor, reward: f64, action: i64, is_done: bool) -> f64 {
        self.steps += 1;
        if !is_done {
            self.events_buf.push(Event {
                observation: obs.copy(),
                reward: reward,
                action: action,
            });
            0.0
        } else {
            // calc values - gamma cumulative reward
            let mut cumreward: f64 = 0.0;
            let mut values: Vec<f64> = vec![0.0; self.events_buf.len()];
            for i in (0..self.events_buf.len()).rev() {
                values[i] = self.events_buf[i].reward + cumreward * GAMMA;
                cumreward = cumreward * GAMMA + self.events_buf[i].reward;
            }
            for i in 0..self.events_buf.len() {
                if self.memory.len() > MEMORY_CAPACITY {
                    self.memory.pop_front();
                }
                self.memory.push_back(MemoryEvent {
                    observation: self.events_buf[i].observation.copy(),
                    action: self.events_buf[i].action,
                    value: values[i],
                })
            }
            self.events_buf.clear();
            if self.memory.len() < BATCH_SIZE {
                return 0.0;
            }
            // sample memory
            //self.optimizer.zero_grad();
            let observations = Tensor::zeros(&[BATCH_SIZE as i64, INPUT_DIMS], kind::FLOAT_CPU);
            let mut actions: Vec<i64> = vec![0; BATCH_SIZE];
            let mut values: Vec<f64> = vec![0.0; BATCH_SIZE];
            let mut rng = rand::thread_rng();
            for i in 0..BATCH_SIZE {
                let index = rng.gen_range(0..self.memory.len());
                let event = self.memory.get(index).unwrap();
                observations
                    .narrow(0, i as i64, 1)
                    .copy_(&event.observation);
                actions[i] = event.action;
                values[i] = event.value;
            }
            let actions_tensor = Tensor::of_slice(&actions);
            let values_tensor = Tensor::of_slice(&values);

            let predictions = self.network.forward_t(&observations, /* train = */ true);
            let predicted_values = predictions
                .gather(1, &actions_tensor.unsqueeze(-1), false)
                .squeeze();
            let diff = &predicted_values - &values_tensor;
            let loss = (&diff * &diff).mean(Kind::Float);
            let loss_scalar = loss.double_value(&[]);
            self.optimizer.backward_step_clip(&loss, 1.0);
            loss_scalar
        }
    }
}
