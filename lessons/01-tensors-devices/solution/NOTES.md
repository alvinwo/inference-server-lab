# Lesson 1 reference-solution notes

Read this only after making a serious attempt and recording your reasoning.

## Design choices

`DevicePolicy.resolve` separates a request from the selected device. Explicit `mps` failure raises
an actionable exception; only `auto` may choose CPU when MPS is unavailable. The `reason` string is
part of startup evidence, not control flow.

`seed_everything` seeds Python and PyTorch. It also seeds MPS when that backend is available. This
improves repeatability inside one environment but cannot guarantee equality across versions,
platforms, devices, or nondeterministic operations.

`affine` validates the contract before calling `torch.matmul(inputs, weight.transpose(0, 1))`.
Transposing the two-dimensional weight changes it from `[output_features, input_features]` to
`[input_features, output_features]`. `torch.matmul` preserves all leading input dimensions; adding
the one-dimensional bias broadcasts over them.

`benchmark_operation` separates warmup from measurement. Each sample synchronizes before its start
and after the operation. The first boundary prevents earlier queued work from entering the sample;
the second prevents asynchronous work from escaping it. Raw samples are retained because one median
cannot reveal variability.

## Common wrong turns

- Using `weight.T` without first requiring a two-dimensional weight hides an ambiguous transpose for
  higher-rank tensors.
- Flattening `[batch, tokens, features]` makes the simple example work but discards useful shape
  meaning and creates unnecessary reshapes.
- Letting explicit MPS silently fall back to CPU makes a benchmark label false.
- Timing only the Python call measures dispatch on asynchronous devices, not completed tensor work.
- Including warmup in the reported samples mixes initialization and steady-state behavior.

## Starter-to-solution map

- `devices.py`: validates the preference, resolves an actual device, and seeds random generators.
- `tensor_ops.py`: validates shape/dtype/device invariants and performs the affine projection.
- `timing.py`: synchronizes supported devices, validates counts, warms up, and records samples.

## Remaining limitations

The policy intentionally supports only CPU and MPS. The timer uses wall-clock measurements and does
not pin CPU threads, control system load, record thermals, or use device-native events. The benchmark
workload is tiny and demonstrates method rather than inference-server performance. Lesson 7 builds a
more complete serving benchmark discipline.
