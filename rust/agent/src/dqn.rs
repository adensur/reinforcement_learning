use rand::Rng;
use std::collections::vec_deque::VecDeque;
use tch::{kind, nn, nn::ModuleT, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::agent::*;

const GAMMA: f64 = 0.999;

const BATCH_SIZE: usize = 128;

#[derive(Debug)]
pub struct DQNNetwork {
    pub input_layer: nn::Linear,
    pub layers: Vec<nn::Linear>,
    pub final_layer: nn::Linear,
    hidden_layers: usize,
    hidden_size: i64,
}

impl DQNNetwork {
    fn new(
        vs: &nn::Path,
        input_dims: i64,
        output_dims: i64,
        hidden_size: i64,
        hidden_layers: usize,
    ) -> Self {
        let input_layer = nn::linear(vs, input_dims, hidden_size, Default::default());
        let mut layers: Vec<nn::Linear> = Vec::new();
        for _ in 0..hidden_layers {
            layers.push(nn::linear(vs, hidden_size, hidden_size, Default::default()));
        }
        let final_layer = nn::linear(vs, hidden_size, output_dims, Default::default());
        Self {
            input_layer: input_layer,
            layers: layers,
            final_layer: final_layer,
            hidden_layers,
            hidden_size,
        }
    }
    fn copy_(&mut self, other: &DQNNetwork) {
        if self.hidden_layers != other.hidden_layers || self.hidden_size != other.hidden_size {
            panic!("!!!");
        }
        tch::no_grad(|| {
            if let Some(bs) = &mut self.input_layer.bs {
                bs.copy_(other.input_layer.bs.as_ref().unwrap());
            }
            self.input_layer.ws.copy_(&other.input_layer.ws);
            if let Some(bs) = &mut self.final_layer.bs {
                bs.copy_(other.final_layer.bs.as_ref().unwrap());
            }
            self.final_layer.ws.copy_(&other.final_layer.ws);
            for i in 0..self.layers.len() {
                if let Some(bs) = &mut self.layers[i].bs {
                    bs.copy_(other.layers[i].bs.as_ref().unwrap());
                }
                self.layers[i].ws.copy_(&other.layers[i].ws);
            }
        });
    }
}

impl nn::ModuleT for DQNNetwork {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let mut t = xs.apply(&self.input_layer).relu();
        for layer in &self.layers {
            t = t.apply(layer).relu();
        }
        t.apply(&self.final_layer)
    }
}

struct TransitionEvent {
    observation: Tensor,
    next_observation: Option<Tensor>,
    reward: f64,
    action: i64,
}

pub struct DQNAgent {
    pub policy_network: DQNNetwork,
    target_network: DQNNetwork,
    optimizer: tch::nn::Optimizer,
    steps: i64,
    input_dims: i64,
    output_dims: i64,
    device: Device,
    memory: VecDeque<TransitionEvent>,
    memory_capacity: usize,
    target_update_freq: i64,
    exploration_prob_decay_steps: i64,
}

impl DQNAgent {
    pub fn new(
        input_dims: i64,
        output_dims: i64,
        memory_capacity: usize,
        target_update_freq: i64,
        exploration_prob_decay_steps: i64,
    ) -> Self {
        let device = Device::cuda_if_available();
        println!("Detected device: {:?}", device);
        let vs = nn::VarStore::new(device);
        let opt = nn::RmsProp::default().build(&vs, 1e-2).unwrap();
        let policy_network = DQNNetwork::new(&vs.root(), input_dims, output_dims, 32, 2);
        let target_network = DQNNetwork::new(&vs.root(), input_dims, output_dims, 32, 2);
        Self {
            policy_network,
            target_network,
            optimizer: opt,
            input_dims,
            output_dims,
            device,
            steps: 0,
            memory: VecDeque::with_capacity(memory_capacity),
            memory_capacity,
            target_update_freq,
            exploration_prob_decay_steps,
        }
    }
}

impl Agent for DQNAgent {
    fn forward(&self, obs: &Tensor) -> Tensor {
        obs.copy()
    }
    fn select_action(&self, obs: &Tensor) -> i64 {
        let obs = obs.to_device(self.device);
        if need_eps_greedy_exploration(self.steps, self.exploration_prob_decay_steps) {
            let mut rng = rand::thread_rng();
            let action = rng.gen_range(0..self.output_dims);
            action
        } else {
            let pred = self
                .policy_network
                .forward_t(&obs.unsqueeze(0), false)
                .squeeze(); // obs is a single tensor, we should expect network to only work on batches
            pred.max_dim(0, false).1.int64_value(&[])
        }
    }
    fn consume_event(
        &mut self,
        obs: &Tensor,
        next_obs: Option<&Tensor>,
        reward: f64,
        action: i64,
        _is_done: bool,
    ) -> f64 {
        self.steps += 1;
        if self.memory.len() > self.memory_capacity {
            self.memory.pop_front();
        }
        self.memory.push_back(TransitionEvent {
            observation: obs.copy(),
            next_observation: next_obs.and_then(|t| Some(t.copy())),
            reward: reward,
            action: action,
        });
        if self.memory.len() < BATCH_SIZE {
            return 0.0;
        }
        // sample memory
        //self.optimizer.zero_grad();
        let float_kind = match self.device {
            Device::Cpu => kind::FLOAT_CPU,
            Device::Cuda(_) => kind::FLOAT_CUDA,
        };
        let observations = Tensor::zeros(&[BATCH_SIZE as i64, self.input_dims], float_kind);
        let mut actions: Vec<i64> = vec![0; BATCH_SIZE];
        let mut rewards: Vec<f64> = vec![0.0; BATCH_SIZE];
        let mut predicted_next_state_values: Vec<f64> = vec![0.0; BATCH_SIZE];
        let mut rng = rand::thread_rng();
        for i in 0..BATCH_SIZE {
            let index = rng.gen_range(0..self.memory.len());
            let transition = self.memory.get(index).unwrap();
            observations
                .narrow(0, i as i64, 1)
                .copy_(&transition.observation);
            actions[i] = transition.action;
            rewards[i] = transition.reward;
            if let Some(next_obs) = &transition.next_observation {
                let pred = tch::no_grad(|| self.target_network.forward_t(next_obs, false));
                predicted_next_state_values[i] = pred.max_dim(0, false).0.double_value(&[]);
            }
        }
        let actions_tensor = Tensor::of_slice(&actions).to_device(self.device);
        let rewards_tensor = Tensor::of_slice(&rewards).to_device(self.device);
        let predicted_next_state_values_tensor =
            Tensor::of_slice(&predicted_next_state_values).to_device(self.device);
        let predictions = self
            .policy_network
            .forward_t(&observations, /* train = */ true);
        let predicted_values = predictions
            .gather(1, &actions_tensor.unsqueeze(-1), false)
            .squeeze();
        // instead of values tensor we have pred(target_net) * GAMMA + rewards
        let expected_state_action_values =
            predicted_next_state_values_tensor * GAMMA + rewards_tensor;
        let diff = &predicted_values - &expected_state_action_values;
        let loss = (&diff * &diff).mean(Kind::Float);
        let loss_scalar = loss.double_value(&[]);
        self.optimizer.backward_step(&loss);
        if self.steps % self.target_update_freq == 0 {
            self.target_network.copy_(&self.policy_network);
        }
        loss_scalar
    }
}
