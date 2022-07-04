use agent::{Agent, DQNAgent, NaiveAgent, PolicyGradientAgent, SimpleAgent};
use gym::GymEnv;
use std::time::Instant;
use structopt::StructOpt;
use strum::VariantNames;
use tch::Kind;

#[derive(Debug, strum::EnumString, strum::EnumVariantNames)]
#[strum(serialize_all = "kebab-case")]
enum Method {
    Simple,
    PolicyGradient,
    DQN,
    Naive,
}

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    #[structopt(short = "e", long = "epochs", default_value = "100000")]
    epochs: usize,
    #[structopt(short = "v", long = "validate-epochs", default_value = "100")]
    validate_epochs: usize,
    #[structopt(short = "c", long = "memory-capacity", default_value = "1000")]
    memory_capacity: usize,
    #[structopt(short = "r", long = "report-freq", default_value = "1000")]
    report_freq: usize,
    #[structopt(short = "r", long = "target-update-freq", default_value = "10")]
    target_update_freq: i64,
    #[structopt(short = "b", long = "batch-size", default_value = "5000")]
    batch_size: usize,
    #[structopt(
        short = "r",
        long = "exploration-prob-decay-steps",
        default_value = "100000"
    )]
    exploration_prob_decay_steps: i64,
    #[structopt(short = "m", long = "method", default_value = "simple", possible_values = Method::VARIANTS)]
    method: Method,
}

fn main() {
    let mut opt = Opt::from_args();
    let env = GymEnv::new("CartPole-v1").unwrap();
    let input_dims = env.observation_space()[0];
    let output_dims = env.action_space();
    let mut agent: Box<dyn Agent> = match opt.method {
        Method::Simple => Box::new(SimpleAgent::new(
            "dense_net",
            input_dims,
            output_dims,
            opt.memory_capacity,
            opt.exploration_prob_decay_steps,
        )),
        Method::PolicyGradient => Box::new(PolicyGradientAgent::new(
            "dense_net",
            input_dims,
            output_dims,
            opt.exploration_prob_decay_steps,
        )),
        Method::DQN => Box::new(DQNAgent::new(
            input_dims,
            output_dims,
            opt.memory_capacity,
            opt.target_update_freq,
            opt.exploration_prob_decay_steps,
        )),
        Method::Naive => Box::new(NaiveAgent::new(
            "dense_net",
            input_dims,
            output_dims,
            opt.batch_size,
            opt.exploration_prob_decay_steps,
        )),
    };
    let mut obs = env.reset().unwrap();
    {
        let pred = agent.forward(&obs).softmax(0, Kind::Float);
        println!("Test network prediction:");
        pred.print();
    }
    //let action_space = env.action_space();
    if opt.report_freq > opt.epochs {
        panic!("!!!");
    }
    let mut episodes: Vec<i64> = Vec::with_capacity(opt.epochs);
    let mut losses: Vec<f64> = Vec::with_capacity(opt.epochs);
    let mut flag = false;
    let start = Instant::now();
    for epoch in 0..opt.epochs {
        let mut i = 0;
        loop {
            let action = agent.select_action(&obs);
            let step = env.step(action).unwrap();
            let is_done = step.is_done;
            let next_obs = match step.is_done {
                true => None,
                false => Some(&step.obs),
            };
            let loss = agent.consume_event(&obs, next_obs, step.reward, action, step.is_done);
            obs = step.obs;
            if is_done || i > 1_000_000 {
                episodes.push(i);
                losses.push(loss);
                obs = env.reset().unwrap();
                break;
            }
            i += 1;
        }
        if flag || (epoch > 0 && epoch % opt.report_freq == 0) {
            opt.report_freq = std::cmp::min(episodes.len(), opt.report_freq);
            let avg_len = episodes[episodes.len() - opt.report_freq..]
                .iter()
                .sum::<i64>() as f64
                / opt.report_freq as f64;
            let max_len = *episodes[episodes.len() - opt.report_freq..]
                .iter()
                .max()
                .unwrap();
            let avg_loss = losses[losses.len() - opt.report_freq..].iter().sum::<f64>()
                / opt.report_freq as f64;
            println!(
                "Epoch {}, last {} epochs avg episode len {}, max_len {}, avg loss {}",
                epoch, opt.report_freq, avg_len, max_len, avg_loss
            );
            // do some validate measurements
            {
                let mut obs = env.reset().unwrap();
                let mut wins_count = 0;
                let mut episodes: Vec<i64> = Vec::with_capacity(opt.epochs);
                for _validate_epoch in 0..opt.validate_epochs {
                    i = 0;
                    loop {
                        let action = agent.select_action(&obs);
                        let step = env.step(action).unwrap();
                        obs = step.obs;
                        let is_done = step.is_done;
                        if is_done {
                            episodes.push(i);
                            obs = env.reset().unwrap();
                            if i >= 499 {
                                wins_count += 1;
                            }
                            break;
                        }
                        i += 1;
                    }
                }
                let avg_len = episodes.iter().sum::<i64>() as f64 / episodes.len() as f64;
                let max_len = *episodes.iter().max().unwrap();
                println!(
                    "Validation results: won {} games out of {}, avg len {}, max len {}",
                    wins_count, opt.validate_epochs, avg_len, max_len,
                );
                if wins_count == opt.validate_epochs {
                    flag = true;
                }
            }
        }
        if flag {
            break;
        }
    }
    let duration = start.elapsed();
    println!(
        "Elapsed seconds: {}, episodes per second: {}, steps per second: {}",
        duration.as_secs(),
        episodes.len() as f64 / duration.as_secs() as f64,
        episodes.iter().sum::<i64>() as f64 / duration.as_secs() as f64
    )
}
