# Contributing to fpo-rust

Thank you for your interest in contributing to fpo-rust! We welcome contributions of all kinds.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fpo-rust.git
   cd fpo-rust
   ```
3. Create a new branch for your changes:
   ```bash
   git checkout -b feature/my-feature
   ```

## Development Workflow

### Prerequisites

- Rust 1.70 or later
- Cargo

### Building

```bash
cargo build
```

### Running Tests

```bash
cargo test
```

### Running the CLI

```bash
cargo run -- run --model cct-s-v2-global-model image.jpg
```

### Checking Code Quality

```bash
# Format code
cargo fmt

# Lint with clippy
cargo clippy -- -D warnings
```

## Commit Guidelines

- Write clear, descriptive commit messages
- Reference issue numbers if applicable: `Fixes #123`
- Keep commits atomic and focused on a single change
- Example: `feat: add support for custom ONNX models`

## Code Style

- Follow the standard Rust style guide
- Run `cargo fmt` before committing
- Use `cargo clippy` to catch common mistakes
- Add documentation comments for public APIs

## Pull Request Process

1. Update documentation and README if needed
2. Add tests for new features
3. Ensure all tests pass: `cargo test`
4. Ensure code is formatted: `cargo fmt`
5. Push your branch and create a Pull Request
6. Describe your changes in the PR description
7. Wait for code review and CI checks to pass

## PR Description Template

```markdown
## Description
Brief description of your changes

## Related Issues
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added
- [ ] Manual testing performed
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] No new warnings from clippy
```

## Reporting Bugs

When reporting bugs, please include:

1. Rust version: `rustc --version`
2. OS and version
3. Steps to reproduce
4. Expected behavior
5. Actual behavior
6. Any error messages or stack traces
7. Model name (if using the CLI)

Example:

```
**Describe the bug**
License plate recognition returns incorrect characters for images at certain angles.

**To Reproduce**
1. Run `fpo-rust run --model cct-s-v2-global-model tilted_plate.jpg`
2. Result shows incorrect characters

**Expected behavior**
Should recognize the plate correctly

**Environment**
- Rust: 1.75.0
- OS: macOS 14.2
- Model: cct-s-v2-global-model
```

## Suggesting Enhancements

We welcome suggestions for improvements! Please:

1. Describe the enhancement clearly
2. Explain the use case
3. Provide examples if possible
4. Consider performance implications

## Areas for Contribution

- **Performance improvements** - Optimize inference speed or memory usage
- **Model support** - Add support for new license plate models
- **Platform support** - Test and fix issues on different platforms
- **Documentation** - Improve docs, add examples
- **Bug fixes** - Report and fix issues
- **Testing** - Improve test coverage

## License

By contributing to fpo-rust, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue with the `question` label or reach out to the maintainers.

Thank you for contributing to fpo-rust! 🙏

