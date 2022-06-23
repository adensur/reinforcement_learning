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
    #[structopt(short = "v", long = "validate-epochs", default_value = "100")]
    validate_epochs: usize,
    #[structopt(short = "r", long = "report-freq", default_value = "1000")]
    report_freq: usize,
    #[structopt(short = "m", long = "method", default_value = "simple", possible_values = Method::VARIANTS)]
    method: Method,
}

fn main() {
    let mut opt = Opt::from_args();
    let env = GymEnv::new("CartPole-v1").unwrap();
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
    let mut losses: Vec<f64> = Vec::with_capacity(opt.epochs);
    let mut flag = false;
    let start = Instant::now();
    for epoch in 0..opt.epochs {
        let mut i = 0;
        loop {
            let action = agent.select_action(&obs);
            let step = env.step(action).unwrap();
            let is_done = step.is_done;
            let loss = agent.consume_event(&obs, step.reward, action, step.is_done);
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
            let mut wins_count = 0;
            for _validate_epoch in 0..opt.validate_epochs {
                let mut validate_episodes: Vec<i64> = Vec::with_capacity(opt.epochs);
                loop {
                    let action = agent.select_action(&obs);
                    let step = env.step(action).unwrap();
                    let is_done = step.is_done;
                    if is_done || i > 1_000_000 {
                        validate_episodes.push(i);
                        obs = env.reset().unwrap();
                        if i >= 499 {
                            wins_count += 1;
                        }
                        break;
                    }
                    i += 1;
                }
            }
            println!(
                "Validation results: won {} games out of {}",
                wins_count, opt.validate_epochs
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
