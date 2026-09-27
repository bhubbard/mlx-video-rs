use thiserror::Error;

#[derive(Error, Debug)]
pub enum VideoError {
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("MLX IO error: {0}")]
    MlxIo(#[from] mlx_rs::error::IoError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("LoRA error: {0}")]
    LoRA(String),

    #[error("Weight error: {0}")]
    Weight(String),
}

pub type Result<T> = std::result::Result<T, VideoError>;
