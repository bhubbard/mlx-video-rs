.PHONY: all build check test coverage clippy fmt format clean info run help

all: check test build

build:
	@cargo build --release

check:
	@cargo check --all-targets

test:
	@cargo test -- --nocapture

coverage:
	@cargo llvm-cov --html
	@echo "Coverage HTML report generated at target/llvm-cov/html/index.html"

coverage-text:
	@cargo llvm-cov

clippy:
	@cargo clippy --all-targets -- -D warnings

fmt:
	@cargo fmt --all -- --check

format:
	@cargo fmt --all

info:
	@cargo run --bin mlx-video -- info

clean:
	@cargo clean

help:
	@echo "MLX-Video-RS (Apple Silicon Metal Accelerated)"
	@echo ""
	@echo "Available targets:"
	@echo "  make build         Build release binary (target/release/mlx-video)"
	@echo "  make check         Check all targets without code generation"
	@echo "  make test          Run all unit tests"
	@echo "  make coverage      Generate LLVM HTML code coverage report"
	@echo "  make coverage-text Print terminal coverage report"
	@echo "  make clippy        Run clippy linter"
	@echo "  make fmt           Check formatting with rustfmt"
	@echo "  make format        Format code with rustfmt"
	@echo "  make info          Display Metal GPU status & model capabilities"
	@echo "  make clean         Clean build artifacts"
