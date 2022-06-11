use gym::GymEnv;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let env = GymEnv::new("CartPole-v1").unwrap();
    let obs = env.reset().unwrap();
    println!("Observation: ");
    obs.print();
    let action_space = env.action_space();
    let mut i = 0;
    loop {
        let action = rng.gen_range(0..action_space);
        let step = env.step(action).unwrap();
        if step.is_done {
            break;
        }
        i += 1;
    }
    println!("Played for {} steps!", i);
}
