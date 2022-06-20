use agent::SimpleAgent;
use gym::GymEnv;
use std::time::Instant;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    #[structopt(short = "e", long = "epochs", default_value = "100000")]
    epochs: usize,
    #[structopt(short = "r", long = "report-freq", default_value = "1000")]
    report_freq: usize,
}

fn main() {
    let mut opt = Opt::from_args();
    let mut agent = SimpleAgent::new("conv_net");
    let env = GymEnv::new("CartPole-v1").unwrap();
    let _ = env.reset().unwrap(); // only use images!
    let mut img = env.render().unwrap().flatten(0, 2);
    //let action_space = env.action_space();
    if opt.report_freq > opt.epochs {
        opt.report_freq = opt.epochs;
    }
    let mut episodes: Vec<i64> = Vec::with_capacity(opt.epochs);
    let mut losses: Vec<f64> = Vec::with_capacity(opt.epochs);
    let mut flag = false;
    let start = Instant::now();
    for epoch in 0..opt.epochs {
        let mut i = 0;
        loop {
            let action = agent.select_action(&img);
            let step = env.step(action).unwrap();
            let is_done = step.is_done;
            let loss = agent.consume_event(&img, step.reward, action, step.is_done);
            img = env.render().unwrap().flatten(0, 2);
            if is_done || i > 1_000_000 {
                episodes.push(i);
                losses.push(loss);
                _ = env.reset().unwrap();
                img = env.render().unwrap().flatten(0, 2);
                if i >= 499 {
                    flag = true;
                }
                break;
            }
            i += 1;
        }
        if flag || (epoch > 0 && epoch % opt.report_freq == 0) {
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
