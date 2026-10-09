# GitHub Upload Checklist

Upload these files and folders to GitHub:

```text
.gitignore
KAGGLE.md
README.md
requirements.txt
pyproject.toml
configs/
docs/
scripts/
src/
tests/
```

Do not upload these folders:

```text
data/
work/
outputs/
__pycache__/
.venv/
```

After upload, your GitHub repo should show `requirements.txt` at the top level. Kaggle will fail if `requirements.txt`, `pyproject.toml`, `scripts/`, or `src/` are missing.

Recommended Kaggle clone cell:

```python
!git clone https://github.com/Smmyak/rsdehamba-lite.git
%cd rsdehamba-lite
```

