use super::config::{STAGE_1_SIGMAS, STAGE_2_SIGMAS};
use crate::error::Result;
use mlx_rs::Array;

pub struct LTXSampler {
    pub sigmas: Vec<f32>,
    pub step_index: usize,
}

impl LTXSampler {
    pub fn for_stage_1() -> Self {
        Self {
            sigmas: STAGE_1_SIGMAS.to_vec(),
            step_index: 0,
        }
    }

    pub fn for_stage_2() -> Self {
        Self {
            sigmas: STAGE_2_SIGMAS.to_vec(),
            step_index: 0,
        }
    }

    pub fn for_dev(steps: usize) -> Self {
        let mut sigmas = Vec::with_capacity(steps + 1);
        for i in 0..steps {
            let t = 1.0 - (i as f32 / steps as f32);
            sigmas.push(t);
        }
        sigmas.push(0.0);
        Self {
            sigmas,
            step_index: 0,
        }
    }

    pub fn current_sigma(&self) -> f32 {
        if self.step_index < self.sigmas.len() {
            self.sigmas[self.step_index]
        } else {
            0.0
        }
    }

    pub fn is_finished(&self) -> bool {
        self.step_index + 1 >= self.sigmas.len()
    }

    pub fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array> {
        let dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index];
        let dt_arr = Array::from_f32(dt);
        let delta = model_output.multiply(&dt_arr)?;
        let next_sample = sample.add(&delta)?;
        self.step_index += 1;
        Ok(next_sample)
    }
}
