use rand::Rng;
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

#[derive(Debug)]
pub struct NaiveNetwork {
    pub input_layer: nn::Linear,
    pub layers: Vec<nn::Linear>,
    pub final_layer: nn::Linear,
}

impl NaiveNetwork {
    fn new(
        vs: &nn::Path,
        input_dims: i64,
        output_dims: i64,
        hidden_size: i64,
        hidden_layers: usize,
    ) -> NaiveNetwork {
        let input_layer = nn::linear(vs, input_dims, hidden_size, Default::default());
        let mut layers: Vec<nn::Linear> = Vec::new();
        for _ in 0..hidden_layers {
            layers.push(nn::linear(vs, hidden_size, hidden_size, Default::default()));
        }
        let final_layer = nn::linear(vs, hidden_size, output_dims, Default::default());
        NaiveNetwork {
            input_layer: input_layer,
            layers: layers,
            final_layer: final_layer,
        }
    }
}

impl nn::ModuleT for NaiveNetwork {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let mut t = xs.apply(&self.input_layer).relu();
        for layer in &self.layers {
            t = t.apply(layer).relu();
        }
        t.apply(&self.final_layer)
    }
}

pub struct NaiveAgent {
    pub network: Box<dyn nn::ModuleT>,
    events_buf: Vec<Event>,
    optimizer: tch::nn::Optimizer,
    steps: i64,
    input_dims: i64,
    output_dims: i64,
    device: Device,
    batch_size: usize,
    exploration_prob_decay_steps: i64,
}

impl NaiveAgent {
    pub fn new(
        network_type: &str,
        input_dims: i64,
        output_dims: i64,
        batch_size: usize,
        exploration_prob_decay_steps: i64,
    ) -> Self {
        let device = Device::cuda_if_available();
        println!("Detected device: {:?}", device);
        let vs = nn::VarStore::new(device);
        let opt = nn::RmsProp::default().build(&vs, 1e-2).unwrap();
        let network: Box<dyn nn::ModuleT> = match network_type {
            "dense_net" => Box::new(NaiveNetwork::new(
                &vs.root(),
                input_dims,
                output_dims,
                32,
                2,
            )),
            _ => panic!(""),
        };
        let mut input_dims = input_dims;
        if network_type == "conv_net" {
            input_dims = 3 * 400 * 600;
        }
        NaiveAgent {
            network,
            events_buf: Vec::new(),
            optimizer: opt,
            steps: 0,
            input_dims,
            output_dims,
            device,
            batch_size,
            exploration_prob_decay_steps,
        }
    }
}

impl Agent for NaiveAgent {
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
        if !is_done && self.events_buf.len() < self.batch_size {
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
            let float_kind = match self.device {
                Device::Cpu => kind::FLOAT_CPU,
                Device::Cuda(_) => kind::FLOAT_CUDA,
            };
            let observations =
                Tensor::zeros(&[self.events_buf.len() as i64, self.input_dims], float_kind);
            let mut actions: Vec<i64> = vec![0; self.events_buf.len()];
            for i in 0..self.events_buf.len() {
                let event = &self.events_buf[i];
                observations
                    .narrow(0, i as i64, 1)
                    .copy_(&event.observation);
                actions[i] = event.action;
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
            self.events_buf.clear();
            loss_scalar
        }
    }
}
