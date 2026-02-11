# Benchmark Models

Download scripts place ONNX models in this directory. Models are not checked
into version control.

## Required Models

| Model | Task | Size | Download |
|-------|------|------|----------|
| MobileNetV2 | Vision (ImageNet) | ~14 MB | `./scripts/download-models.sh` |
| DistilBERT | Text (NLI) | ~256 MB | `./scripts/download-models.sh` |
| Whisper-tiny | Audio/Signal | ~151 MB | `./scripts/download-models.sh` |

## File Layout

After download:

```
models/
  mobilenetv2-7.onnx
  distilbert-base-uncased.onnx
  whisper-tiny-encoder.onnx
```
