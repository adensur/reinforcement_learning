use agent::{Agent, PolicyGradientAgent, SimpleAgent};
use gym::GymEnv;
use std::time::Instant;
use structopt::StructOpt;
use strum::VariantNames;
use tch::{kind, nn, nn::OptimizerConfig, Device, Kind, Tensor};

#[derive(Debug, strum::EnumString, strum::EnumVariantNames)]
#[strum(serialize_all = "kebab-case")]
enum Method {
    Simple,
    PolicyGradient,
}

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    #[structopt(short = "e", long = "epochs", default_value = "100000")]
    epochs: usize,
    #[structopt(short = "r", long = "report-freq", default_value = "1000")]
    report_freq: usize,
    #[structopt(short = "m", long = "method", default_value = "simple", possible_values = Method::VARIANTS)]
    method: Method,
}

/*
    https://www.gymlibrary.ml/environments/box2d/lunar_lander/
    The env is solved when total reward is 200
*/

fn main() {
    let mut opt = Opt::from_args();
    let env = GymEnv::new("LunarLander-v2").unwrap();
    let input_dims = env.observation_space()[0];
    let output_dims = env.action_space();
    let mut agent: Box<dyn Agent> = match opt.method {
        Method::Simple => Box::new(SimpleAgent::new("dense_net", input_dims, output_dims)),
        Method::PolicyGradient => Box::new(PolicyGradientAgent::new(
            "dense_net",
            input_dims,
            output_dims,
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
    let mut total_rewards: Vec<f64> = Vec::with_capacity(opt.epochs);
    let mut losses: Vec<f64> = Vec::with_capacity(opt.epochs);
    let mut flag = false;
    let start = Instant::now();
    let mut total_reward = 0.0;
    let mut wins_count = 0;
    for epoch in 0..opt.epochs {
        let mut i = 0;
        loop {
            let action = agent.select_action(&obs);
            let step = env.step(action).unwrap();
            total_reward += step.reward;
            let is_done = step.is_done;
            let loss = agent.consume_event(&obs, step.reward, action, step.is_done);
            obs = step.obs;
            if is_done || i > 1_000_000 {
                episodes.push(i);
                losses.push(loss);
                total_rewards.push(total_reward);
                obs = env.reset().unwrap();
                if total_reward >= 200.0 {
                    wins_count += 1;
                    println!("We won!!!! Wins in a row: {}", wins_count);
                    if wins_count >= 10 {
                        flag = true;
                    }
                } else {
                    wins_count = 0;
                }
                total_reward = 0.0;
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
            let avg_total_reward = total_rewards[total_rewards.len() - opt.report_freq..]
                .iter()
                .sum::<f64>()
                / opt.report_freq as f64;
            let max_total_reward = total_rewards[total_rewards.len() - opt.report_freq..]
                .iter()
                .map(|x| *x)
                .fold(f64::NEG_INFINITY, f64::max);
            let avg_loss = losses[losses.len() - opt.report_freq..].iter().sum::<f64>()
                / opt.report_freq as f64;
            println!(
                "Epoch {}, last {} epochs avg episode len {}, max_len {}, avg total reward {}, max total reward {}, avg loss {}",
                epoch, opt.report_freq, avg_len, max_len, avg_total_reward, max_total_reward, avg_loss
            )
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
