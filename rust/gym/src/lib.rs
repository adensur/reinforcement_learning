//! Wrappers around the Python API of the OpenAI gym.
use cpython::{NoArgs, ObjectProtocol, PyDict, PyObject, PyResult, Python, ToPyObject};
use tch::Tensor;

/// The return value for a step.
#[derive(Debug)]
pub struct Step<A> {
    pub obs: Tensor,
    pub action: A,
    pub reward: f64,
    pub is_done: bool,
}

impl<A: Copy> Step<A> {
    /// Returns a copy of this step changing the observation tensor.
    pub fn copy_with_obs(&self, obs: &Tensor) -> Step<A> {
        Step {
            obs: obs.copy(),
            action: self.action,
            reward: self.reward,
            is_done: self.is_done,
        }
    }
}

/// An OpenAI Gym session.
pub struct GymEnv {
    env: PyObject,
    action_space: i64,
    observation_space: Vec<i64>,
}

impl GymEnv {
    /// Creates a new session of the specified OpenAI Gym environment.
    pub fn new(name: &str) -> PyResult<GymEnv> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let sys = py.import("sys").unwrap();
        let version: String = sys.get(py, "version")?.extract(py)?;
        println!("Python version: {}", version);
        let gym = py.import("gym")?;
        let gym_version: String = gym.get(py, "__version__")?.extract(py)?;
        println!("Gym version: {}", gym_version);
        let env = gym.call(py, "make", (name,), None)?;
        //let _ = env.call_method(py, "seed", (42,), None)?;
        let action_space = env.getattr(py, "action_space")?;
        let action_space = if let Ok(val) = action_space.getattr(py, "n") {
            val.extract(py)?
        } else {
            let action_space: Vec<i64> = action_space.getattr(py, "shape")?.extract(py)?;
            action_space[0]
        };
        let observation_space = env.getattr(py, "observation_space")?;
        let observation_space = observation_space.getattr(py, "shape")?.extract(py)?;
        Ok(GymEnv {
            env,
            action_space,
            observation_space,
        })
    }

    /// Resets the environment, returning the observation tensor.
    pub fn reset(&self) -> PyResult<Tensor> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let obs = self.env.call_method(py, "reset", NoArgs, None)?;
        Ok(Tensor::of_slice(&obs.extract::<Vec<f32>>(py)?))
    }

    /// Applies an environment step using the specified action.
    pub fn step<A: ToPyObject + Copy>(&self, action: A) -> PyResult<Step<A>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let step = self.env.call_method(py, "step", (action,), None)?;
        Ok(Step {
            obs: Tensor::of_slice(&step.get_item(py, 0)?.extract::<Vec<f32>>(py)?),
            reward: step.get_item(py, 1)?.extract(py)?,
            is_done: step.get_item(py, 2)?.extract(py)?,
            action,
        })
    }

    pub fn render(&self) -> PyResult<Tensor> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let args = PyDict::new(py);
        args.set_item(py, "mode", "rgb_array")?;
        let img = self
            .env
            .call_method(py, "render", NoArgs, Some(&args))
            .unwrap();
        let flat_img = img
            .call_method(py, "flatten", NoArgs, None)
            .unwrap()
            .extract::<Vec<f32>>(py)
            .unwrap();
        let shape_vec = img
            .getattr(py, "shape")
            .unwrap()
            .extract::<Vec<i64>>(py)
            .unwrap();
        let mut shape = [0, 0, 0];
        if shape_vec.len() != shape.len() {
            panic!("");
        }
        for i in 0..shape.len() {
            shape[i] = shape_vec[i];
        }
        let t = Tensor::of_slice(&flat_img).view(shape).permute(&[2, 0, 1]);
        Ok(t)
    }

    /// Returns the number of allowed actions for this environment.
    pub fn action_space(&self) -> i64 {
        self.action_space
    }

    /// Returns the shape of the observation tensors.
    pub fn observation_space(&self) -> &[i64] {
        &self.observation_space
    }
}
