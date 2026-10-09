# Kaggle Setup

Use this after you upload the full project to GitHub.

## 1. Create Notebook

1. Open Kaggle.
2. Create a new notebook.
3. Turn on GPU from notebook settings.
4. Click **Add Input** and add the SateHaze1K dataset.

Dataset to search:

```text
xuxingxing233/satehaze1k
```

## 2. Clone Project

Replace the URL if your GitHub repo name changes.

```python
!git clone https://github.com/Smmyak/rsdehamba-lite.git
%cd rsdehamba-lite
```

## 3. Install

```python
!pip install -q -r requirements.txt
!pip install -q -e .
```

## 4. Check Dataset Folder Name

```python
from pathlib import Path

for root in Path("/kaggle/input").iterdir():
    print(root)
    for child in list(root.iterdir())[:10]:
        print("  ", child.name)
```

Set `DATA_ROOT` to the correct folder:

```python
DATA_ROOT = "/kaggle/input/satehaze1k"
```

If Kaggle uses a different name, change the path.

## 5. Inspect Pairs

```python
!python scripts/inspect_dataset.py --data-root "$DATA_ROOT"
```

If it prints `pairs: 0`, stop and share the folder listing. The loader needs the actual folder names.

## 6. Run Tests And Dry Run

```python
!pytest -q
!python scripts/train.py --config configs/rsdehamba_lite_smoke.json --data-root "$DATA_ROOT" --dry-run
```

## 7. Smoke Train

```python
!python scripts/train.py \
  --config configs/rsdehamba_lite_smoke.json \
  --data-root "$DATA_ROOT" \
  --output-dir /kaggle/working/rsdehamba_lite_outputs
```

## 8. Smoke Evaluate

```python
!python scripts/evaluate.py \
  --config configs/rsdehamba_lite_smoke.json \
  --data-root "$DATA_ROOT" \
  --checkpoint /kaggle/working/rsdehamba_lite_outputs/checkpoints/best.pt
```

## 9. Full Train

Only run this after the smoke train works.

```python
!python scripts/train.py \
  --config configs/rsdehamba_lite.json \
  --data-root "$DATA_ROOT" \
  --output-dir /kaggle/working/rsdehamba_lite_full \
  --epochs 50 \
  --batch-size 4 \
  --image-size 256
```

## 10. Full Evaluate

```python
!python scripts/evaluate.py \
  --config configs/rsdehamba_lite.json \
  --data-root "$DATA_ROOT" \
  --checkpoint /kaggle/working/rsdehamba_lite_full/checkpoints/best.pt
```

