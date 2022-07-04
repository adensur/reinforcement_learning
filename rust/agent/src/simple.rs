use rand::Rng;
use std::collections::vec_deque::VecDeque;
use tch::{kind, nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::agent::*;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

const BATCH_SIZE: usize = 128;

#[derive(Debug)]
pub struct DenseNetwork {
    pub input_layer: nn::Linear,
    pub layers: Vec<nn::Linear>,
    pub final_layer: nn::Linear,
}

impl DenseNetwork {
    fn new(
        vs: &nn::Path,
        input_dims: i64,
        output_dims: i64,
        hidden_size: i64,
        hidden_layers: usize,
    ) -> DenseNetwork {
        let input_layer = nn::linear(vs, input_dims, hidden_size, Default::default());
        let mut layers: Vec<nn::Linear> = Vec::new();
        for _ in 0..hidden_layers {
            layers.push(nn::linear(vs, hidden_size, hidden_size, Default::default()));
        }
        let final_layer = nn::linear(vs, hidden_size, output_dims, Default::default());
        DenseNetwork {
            input_layer: input_layer,
            layers: layers,
            final_layer: final_layer,
        }
    }
}

impl nn::ModuleT for DenseNetwork {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let mut t = xs.apply(&self.input_layer).relu();
        for layer in &self.layers {
            t = t.apply(layer).relu();
        }
        t.apply(&self.final_layer)
    }
}

#[derive(Debug)]
pub struct ConvNetwork {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    conv3: nn::Conv2D,
    bn3: nn::BatchNorm,
    final_layer: nn::Linear,
}

impl ConvNetwork {
    fn new(vs: &nn::Path, output_dims: i64) -> ConvNetwork {
        let mut conv_config = nn::ConvConfig::default();
        conv_config.stride = 2;
        ConvNetwork {
            conv1: nn::conv2d(vs, 3, 16, 5, conv_config),
            conv2: nn::conv2d(vs, 16, 32, 5, conv_config),
            conv3: nn::conv2d(vs, 32, 32, 5, conv_config),
            bn1: nn::batch_norm2d(vs, 16, Default::default()),
            bn2: nn::batch_norm2d(vs, 32, Default::default()),
            bn3: nn::batch_norm2d(vs, 32, Default::default()),
            final_layer: nn::linear(vs, 108288, output_dims, Default::default()),
        }
    }
}

impl nn::ModuleT for ConvNetwork {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // xs is expected to be a flat tensor
        let mut x = xs.view([-1, 3, 400, 600]);
        x = self.bn1.forward_t(&x.apply(&self.conv1), train).relu();
        x = self.bn2.forward_t(&x.apply(&self.conv2), train).relu();
        x = self.bn3.forward_t(&x.apply(&self.conv3), train).relu();
        x = x.flatten(1, 3); // flatten height, widths, channels; leave only batch dimension + 1 flat tensor
        x = x.apply(&self.final_layer);
        x
    }
}

pub struct SimpleAgent {
    pub network: Box<dyn nn::ModuleT>,
    events_buf: Vec<Event>,
    optimizer: tch::nn::Optimizer,
    steps: i64,
    memory: VecDeque<MemoryEvent>,
    input_dims: i64,
    output_dims: i64,
    device: Device,
    memory_capacity: usize,
    exploration_prob_decay_steps: i64,
}

impl SimpleAgent {
    pub fn new(
        network_type: &str,
        input_dims: i64,
        output_dims: i64,
        memory_capacity: usize,
        exploration_prob_decay_steps: i64,
    ) -> Self {
        let device = Device::cuda_if_available();
        println!("Detected device: {:?}", device);
        let vs = nn::VarStore::new(device);
        let opt = nn::RmsProp::default().build(&vs, 1e-2).unwrap();
        let network: Box<dyn nn::ModuleT> = match network_type {
            "dense_net" => Box::new(DenseNetwork::new(
                &vs.root(),
                input_dims,
                output_dims,
                32,
                2,
            )),
            "conv_net" => Box::new(ConvNetwork::new(&vs.root(), output_dims)),
            _ => panic!(""),
        };
        let mut input_dims = input_dims;
        if network_type == "conv_net" {
            input_dims = 3 * 400 * 600;
        }
        SimpleAgent {
            network,
            events_buf: Vec::new(),
            optimizer: opt,
            steps: 0,
            memory: VecDeque::with_capacity(memory_capacity),
            input_dims,
            output_dims,
            device,
            memory_capacity,
            exploration_prob_decay_steps,
        }
    }
}

impl Agent for SimpleAgent {
    fn forward(&self, obs: &Tensor) -> Tensor {
        self.network.forward_t(&obs.unsqueeze(0), false).squeeze()
    }

    fn select_action(&self, obs: &Tensor) -> i64 {
        let obs = obs.to_device(self.device);
        if need_eps_greedy_exploration(self.steps, self.exploration_prob_decay_steps) {
            let mut rng = rand::thread_rng();
            let action = rng.gen_range(0..self.output_dims);
            action
        } else {
            let pred = self.network.forward_t(&obs.unsqueeze(0), false).squeeze(); // obs is a single tensor, we should expect network to only work on batches
            pred.max_dim(0, false).1.int64_value(&[])
        }
    }

    fn consume_event(
        &mut self,
        obs: &Tensor,
        _next_obs: Option<&Tensor>,
        reward: f64,
        action: i64,
        is_done: bool,
    ) -> f64 {
        self.steps += 1;
        if !is_done {
            self.events_buf.push(Event {
                observation: obs.copy(),
                reward: reward,
                action: action,
                is_done,
            });
            0.0
        } else {
            // calc values - gamma cumulative reward
            let values = calc_values(&self.events_buf);
            for (i, event) in std::mem::take(&mut self.events_buf).into_iter().enumerate() {
                if self.memory.len() > self.memory_capacity {
                    self.memory.pop_front();
                }
                self.memory.push_back(MemoryEvent {
                    action: event.action,
                    observation: event.observation,
                    value: values[i],
                })
            }
            self.events_buf.clear();
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
            let actions_tensor = Tensor::of_slice(&actions).to_device(self.device);
            let values_tensor = Tensor::of_slice(&values).to_device(self.device);

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
