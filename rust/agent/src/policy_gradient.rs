use rand::Rng;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

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
pub struct SoftmaxNetwork {
    pub input_layer: nn::Linear,
    pub final_layer: nn::Linear,
}

impl SoftmaxNetwork {
    fn new(
        vs: &nn::Path,
        input_dims: i64,
        output_dims: i64,
        hidden_size: i64,
        _hidden_layers: usize,
    ) -> SoftmaxNetwork {
        let input_layer = nn::linear(vs, input_dims, hidden_size, Default::default());
        let final_layer = nn::linear(vs, hidden_size, output_dims, Default::default());
        SoftmaxNetwork {
            input_layer: input_layer,
            final_layer: final_layer,
        }
    }
}

impl nn::ModuleT for SoftmaxNetwork {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let t = xs.apply(&self.input_layer).tanh();
        t.apply(&self.final_layer)
    }
}

#[derive(Debug)]
pub struct ConvSoftmaxNetwork {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    conv3: nn::Conv2D,
    bn3: nn::BatchNorm,
    final_layer: nn::Linear,
}

impl ConvSoftmaxNetwork {
    fn new(vs: &nn::Path, output_dims: i64) -> ConvSoftmaxNetwork {
        let mut conv_config = nn::ConvConfig::default();
        conv_config.stride = 2;
        ConvSoftmaxNetwork {
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

impl nn::ModuleT for ConvSoftmaxNetwork {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // xs is expected to be a flat tensor
        let mut x = xs.view([-1, 3, 400, 600]);
        x = self.bn1.forward_t(&x.apply(&self.conv1), train).tanh();
        x = self.bn2.forward_t(&x.apply(&self.conv2), train).tanh();
        x = self.bn3.forward_t(&x.apply(&self.conv3), train).tanh();
        x = x.flatten(1, 3); // flatten height, widths, channels; leave only batch dimension + 1 flat tensor
        x = x.apply(&self.final_layer);
        x
    }
}

pub struct PolicyGradientAgent {
    pub network: Box<dyn nn::ModuleT>,
    events_buf: Vec<Event>,
    optimizer: tch::nn::Optimizer,
    steps: i64,
    output_dims: i64,
    device: Device,
    exploration_prob_decay_steps: i64,
}

impl PolicyGradientAgent {
    pub fn new(
        network_type: &str,
        input_dims: i64,
        output_dims: i64,
        exploration_prob_decay_steps: i64,
    ) -> Self {
        let device = Device::cuda_if_available();
        println!("Detected device: {:?}", device);
        let vs = nn::VarStore::new(device);
        let opt = nn::RmsProp::default().build(&vs, 1e-2).unwrap();
        let mut input_dims = input_dims;
        if network_type == "conv_net" {
            input_dims = 3 * 400 * 600;
        }
        let network: Box<dyn nn::ModuleT> = match network_type {
            "dense_net" => Box::new(SoftmaxNetwork::new(
                &vs.root(),
                input_dims,
                output_dims,
                32,
                2,
            )),
            "conv_net" => Box::new(ConvSoftmaxNetwork::new(&vs.root(), output_dims)),
            _ => panic!(""),
        };
        PolicyGradientAgent {
            network,
            events_buf: Vec::new(),
            optimizer: opt,
            steps: 0,
            output_dims,
            device,
            exploration_prob_decay_steps,
        }
    }
}

impl Agent for PolicyGradientAgent {
    fn select_action(&self, obs: &Tensor) -> i64 {
        let obs = obs.to_device(self.device);
        if need_eps_greedy_exploration(self.steps, self.exploration_prob_decay_steps) {
            let mut rng = rand::thread_rng();
            let action = rng.gen_range(0..=1);
            action
        } else {
            let pred = self.network.forward_t(&obs.unsqueeze(0), false).squeeze(); // obs is a single tensor, we should expect network to only work on batches
            let action = pred.softmax(0, Kind::Float).multinomial(1, true);
            let action = i64::from(action);
            action
        }
    }

    fn forward(&self, obs: &Tensor) -> Tensor {
        self.network.forward_t(&obs.unsqueeze(0), false).squeeze()
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
        if !is_done || self.events_buf.len() > 5000 {
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
            let values = Tensor::of_slice(&values).to_device(self.device);
            // apply model
            let observations: Vec<Tensor> = self
                .events_buf
                .iter()
                .map(|s| s.observation.copy())
                .collect();
            let observations: Tensor = Tensor::stack(&observations, 0).to_device(self.device);
            let logits = self.network.forward_t(&observations, /* train = */ true);
            // get actions
            let actions: Vec<i64> = self.events_buf.iter().map(|s| s.action).collect();
            let actions = Tensor::of_slice(&actions).unsqueeze(1);
            let action_mask = Tensor::zeros(
                &[self.events_buf.len() as i64, self.output_dims],
                tch::kind::FLOAT_CPU,
            )
            .scatter_value(1, &actions, 1.0);
            let log_probs = (action_mask * logits.log_softmax(1, Kind::Float)).sum_dim_intlist(
                &[1],
                false,
                Kind::Float,
            );
            let loss = -(values * log_probs).mean(Kind::Float);
            self.optimizer.backward_step(&loss);
            self.events_buf.clear();
            let loss_scalar = loss.double_value(&[]);
            loss_scalar
        }
    }
}
