use rand::Rng;
use std::collections::vec_deque::VecDeque;
use std::f64::consts;
use tch::{kind, nn, nn::OptimizerConfig, Device, Kind, Tensor};

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
pub struct DenseNetwork {
    pub input_layer: nn::Linear,
    pub layers: Vec<nn::Linear>,
    pub final_layer: nn::Linear,
}

impl DenseNetwork {
    fn new(vs: &nn::Path, hidden_size: i64, hidden_layers: usize) -> DenseNetwork {
        let input_layer = nn::linear(vs, INPUT_DIMS, hidden_size, Default::default());
        let mut layers: Vec<nn::Linear> = Vec::new();
        for _ in 0..hidden_layers {
            layers.push(nn::linear(vs, hidden_size, hidden_size, Default::default()));
        }
        let final_layer = nn::linear(vs, hidden_size, OUTPUT_DIMS, Default::default());
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
    fn new(vs: &nn::Path) -> ConvNetwork {
        let mut conv_config = nn::ConvConfig::default();
        conv_config.stride = 2;
        ConvNetwork {
            conv1: nn::conv2d(vs, 3, 16, 5, conv_config),
            conv2: nn::conv2d(vs, 16, 32, 5, conv_config),
            conv3: nn::conv2d(vs, 32, 32, 5, conv_config),
            bn1: nn::batch_norm2d(vs, 16, Default::default()),
            bn2: nn::batch_norm2d(vs, 32, Default::default()),
            bn3: nn::batch_norm2d(vs, 32, Default::default()),
            final_layer: nn::linear(vs, 108288, OUTPUT_DIMS, Default::default()),
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
    pub network: Box<dyn nn::ModuleT>,
    events_buf: Vec<Event>,
    optimizer: tch::nn::Optimizer,
    steps: i64,
    memory: VecDeque<MemoryEvent>,
    input_dims: i64,
    device: Device,
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
    pub fn new(network_type: &str) -> Self {
        let device = Device::cuda_if_available();
        println!("Detected device: {:?}", device);
        let vs = nn::VarStore::new(device);
        let opt = nn::RmsProp::default().build(&vs, 1e-2).unwrap();
        let network: Box<dyn nn::ModuleT> = match network_type {
            "dense_net" => Box::new(DenseNetwork::new(&vs.root(), 32, 2)),
            "conv_net" => Box::new(ConvNetwork::new(&vs.root())),
            _ => panic!(""),
        };
        let mut input_dims = INPUT_DIMS;
        if network_type == "conv_net" {
            input_dims = 3 * 400 * 600;
        }
        SimpleAgent {
            network,
            events_buf: Vec::new(),
            optimizer: opt,
            steps: 0,
            memory: VecDeque::with_capacity(MEMORY_CAPACITY),
            input_dims,
            device,
        }
    }

    pub fn select_action(&self, obs: &Tensor) -> i64 {
        let obs = obs.to_device(self.device);
        let mut rng = rand::thread_rng();
        let rand: f64 = rng.gen_range(0.0..1.0);
        let explore_prob = EXPLORATION_END_PROB
            + (EXPLORATION_START_PROB - EXPLORATION_END_PROB)
                * (consts::E).powf(-1.0f64 * self.steps as f64 / EXPLORATION_PROB_DECAY_STEPS);
        if rand <= explore_prob {
            let action = rng.gen_range(0..=1);
            action
        } else {
            let pred = self.network.forward_t(&obs.unsqueeze(0), false).squeeze(); // obs is a single tensor, we should expect network to only work on batches
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
