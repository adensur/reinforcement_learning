fn need_eps_greedy_exploration(steps: i64, exploration_prob_decay_steps: i64) -> bool {
    let mut rng = rand::thread_rng();
    let rand: f64 = rng.gen_range(0.0..1.0);
    let explore_prob = EXPLORATION_END_PROB
        + (EXPLORATION_START_PROB - EXPLORATION_END_PROB)
            * (consts::E).powf(-1.0f64 * steps as f64 / exploration_prob_decay_steps as f64);
    if rand <= explore_prob {
        true
    } else {
        false
    }
}
