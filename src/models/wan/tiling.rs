#[derive(Debug, Clone, Copy)]
pub struct TilingConfig {
    pub tile_size_t: usize,
    pub tile_size_h: usize,
    pub tile_size_w: usize,
    pub overlap_t: usize,
    pub overlap_h: usize,
    pub overlap_w: usize,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            tile_size_t: 16,
            tile_size_h: 32,
            tile_size_w: 32,
            overlap_t: 4,
            overlap_h: 8,
            overlap_w: 8,
        }
    }
}

impl TilingConfig {
    pub fn new(tile_size: usize, overlap: usize) -> Self {
        Self {
            tile_size_t: tile_size / 2,
            tile_size_h: tile_size,
            tile_size_w: tile_size,
            overlap_t: overlap / 2,
            overlap_h: overlap,
            overlap_w: overlap,
        }
    }
}
