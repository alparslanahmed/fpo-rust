# Publishing to crates.io

This guide explains how to publish `fpo-rust` to crates.io.

## Prerequisites

1. Create an account on [crates.io](https://crates.io)
2. Generate an API token from your account settings
3. Login to crates.io via cargo:
   ```bash
   cargo login
   ```
   Then paste your API token when prompted.

## Publishing Steps

### 1. Update Version

Update the version in `Cargo.toml`:

```toml
[package]
version = "0.1.0"
```

### 2. Update CHANGELOG.md

Add a new section for the version with all notable changes.

### 3. Commit Changes

```bash
git add Cargo.toml CHANGELOG.md
git commit -m "Bump version to 0.1.0"
git push
```

### 4. Test Publishing (Dry Run)

```bash
cargo publish --dry-run --allow-dirty
```

This will:
- Package the crate
- Verify it compiles
- Show what would be uploaded

### 5. Publish to crates.io

```bash
cargo publish --allow-dirty
```

This will upload the crate to crates.io. It typically takes a few minutes for the package to appear on the crates.io website.

### 6. Verify

Visit `https://crates.io/crates/fpo-rust/` to verify the publication.

## Import in Projects

Once published, users can import `fpo-rust` like this:

```toml
[dependencies]
fpo-rust = "0.1"
```

Or pin to a specific version:

```toml
[dependencies]
fpo-rust = "0.1.0"
```

## Documentation

After publishing, API documentation is automatically generated and available at:
`https://docs.rs/fpo-rust/`

## Important Notes

- **Do not publish pre-release or experimental code**
- Always run tests before publishing: `cargo test`
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update README with any new features
- Ensure all examples in README are working

## Troubleshooting

### "crate name already exists"
- The crate name is already taken on crates.io
- Choose a different name and update `Cargo.toml`

### "Package size too large"
- Exclude unnecessary files in `Cargo.toml`:
  ```toml
  exclude = ["tests/", "examples/", ".github/"]
  ```

### "API token not recognized"
- Re-run `cargo login` and enter your token again
- Check your token hasn't expired on crates.io

## Support

For more information, see the [Cargo book on publishing](https://doc.rust-lang.org/cargo/publishing/index.html).

