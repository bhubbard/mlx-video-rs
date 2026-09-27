use crate::error::Result;
use mlx_rs::Array;

/// Compute shifted sigma schedule matching official Wan2.2 scheduler.
pub fn compute_sigmas(num_steps: usize, shift: f32, num_train_timesteps: usize) -> Vec<f32> {
    let n = num_train_timesteps as f64;
    let sigma_max = (n - 1.0) / n;
    let sigma_min = 0.0;

    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for i in 0..num_steps {
        let t = sigma_max + (sigma_min - sigma_max) * (i as f64 / num_steps as f64);
        let s = (shift as f64 * t) / (1.0 + (shift as f64 - 1.0) * t);
        sigmas.push(s as f32);
    }
    sigmas.push(0.0);
    sigmas
}

/// 1st-order Euler scheduler for flow matching diffusion.
#[derive(Debug, Clone)]
pub struct FlowMatchEulerScheduler {
    pub num_train_timesteps: usize,
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
    pub step_index: usize,
}

impl FlowMatchEulerScheduler {
    pub fn new(num_train_timesteps: usize) -> Self {
        Self {
            num_train_timesteps,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            step_index: 0,
        }
    }

    pub fn set_timesteps(&mut self, num_steps: usize, shift: f32) {
        self.sigmas = compute_sigmas(num_steps, shift, self.num_train_timesteps);
        self.timesteps = self.sigmas[..num_steps]
            .iter()
            .map(|&s| (s * self.num_train_timesteps as f32).round())
            .collect();
        self.step_index = 0;
    }

    pub fn reset(&mut self) {
        self.step_index = 0;
    }

    pub fn current_timestep(&self) -> f32 {
        if self.step_index < self.timesteps.len() {
            self.timesteps[self.step_index]
        } else {
            0.0
        }
    }

    /// Euler step: x_next = x + (sigma_next - sigma_cur) * v
    pub fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array> {
        let dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index];
        let dt_arr = Array::from_f32(dt);
        let delta = model_output.multiply(&dt_arr)?;
        let x_next = sample.add(&delta)?;
        self.step_index += 1;
        Ok(x_next)
    }
}

/// DPM-Solver++(2M) for flow matching diffusion with 2nd-order correction.
#[derive(Debug, Clone)]
pub struct FlowDPMPP2MScheduler {
    pub num_train_timesteps: usize,
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
    pub step_index: usize,
    prev_model_output: Option<Array>,
}

impl FlowDPMPP2MScheduler {
    pub fn new(num_train_timesteps: usize) -> Self {
        Self {
            num_train_timesteps,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            step_index: 0,
            prev_model_output: None,
        }
    }

    pub fn set_timesteps(&mut self, num_steps: usize, shift: f32) {
        self.sigmas = compute_sigmas(num_steps, shift, self.num_train_timesteps);
        self.timesteps = self.sigmas[..num_steps]
            .iter()
            .map(|&s| (s * self.num_train_timesteps as f32).round())
            .collect();
        self.step_index = 0;
        self.prev_model_output = None;
    }

    pub fn reset(&mut self) {
        self.step_index = 0;
        self.prev_model_output = None;
    }

    pub fn current_timestep(&self) -> f32 {
        if self.step_index < self.timesteps.len() {
            self.timesteps[self.step_index]
        } else {
            0.0
        }
    }

    pub fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array> {
        let sigma_cur = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma_cur;

        let x_next = if self.step_index > 0 && self.prev_model_output.is_some() {
            let prev_out = match self.prev_model_output.as_ref() {
                Some(p) => p,
                None => unreachable!(),
            };
            let sigma_prev = self.sigmas[self.step_index - 1];
            let h = dt;
            let h_prev = sigma_cur - sigma_prev;
            let r = h / h_prev;

            // D = (1 + 1/(2r)) * v - 1/(2r) * v_prev
            let factor1 = Array::from_f32(1.0 + 0.5 / r);
            let factor2 = Array::from_f32(0.5 / r);
            let term1 = model_output.multiply(&factor1)?;
            let term2 = prev_out.multiply(&factor2)?;
            let d = term1.subtract(&term2)?;

            let h_arr = Array::from_f32(h);
            let delta = d.multiply(&h_arr)?;
            sample.add(&delta)?
        } else {
            // First step: 1st order Euler
            let dt_arr = Array::from_f32(dt);
            let delta = model_output.multiply(&dt_arr)?;
            sample.add(&delta)?
        };

        self.prev_model_output = Some(model_output.clone());
        self.step_index += 1;
        Ok(x_next)
    }
}
