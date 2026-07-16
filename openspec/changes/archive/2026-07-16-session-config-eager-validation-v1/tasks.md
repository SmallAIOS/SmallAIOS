# Tasks — session-config-eager-validation-v1

## 1. Validation authority

- [x] 1.1 Add `SessionConfig::validate(&self) -> Result<(), SessionError>` in `onnx-rt/src/session.rs`, feature-independent, matching `stream_config` and rejecting `Overlap { transfer_streams > 2 }` with `SessionError::InvalidConfig`
- [x] 1.2 Unit-test `validate`: rejects `transfer_streams = 3/5`, accepts `SingleStream` and `transfer_streams = 0/1/2`; test compiles/runs in both default and `cuda` feature sets

## 2. Fallible constructor

- [x] 2.1 Change `Session::new(config) -> Result<Self, SessionError>` to call `config.validate()?` before constructing; update its doc comment
- [x] 2.2 Add a comment at `ensure_stream_pool`'s `transfer_streams` check noting construction is the primary gate and this is a backstop

## 3. Update callers

- [x] 3.1 Update `container/src/main.rs` `Session::new` call sites to propagate the error (`?` / mapped into the binary's error path)
- [x] 3.2 Update `onnx-rt/tests/*` call sites (integration_inference, test_real_model, test_loop_fixture, bench_vision_models, test_cuda) to `.unwrap()`/`?` as fits each test
- [x] 3.3 Grep for any remaining `Session::new(` call sites workspace-wide and update them (examples, benches)

## 4. Quality gates

- [x] 4.1 `just fmt-check` and `just clippy` clean
- [x] 4.2 `just test` green; `cargo test -p smallaios-onnx-rt` green (default + `cuda` where buildable)
- [x] 4.3 `cargo semver-checks` flags the `new` signature break as expected (PR title carries `!`) (API Semver Check gate passed conditionally on #229's `feat(onnx-rt)!:` title)

## 5. Land

- [x] 5.1 `openspec validate session-config-eager-validation-v1 --type change --strict` passes
- [x] 5.2 PR against `develop` titled `feat(onnx-rt)!: validate SessionConfig eagerly at Session::new (session-config-eager-validation-v1)`, closing #127 (landed as #229, merged 2026-07-03)
