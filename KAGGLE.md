# Run the MAI verification pipeline on Kaggle

Yes, this pipeline can run on Kaggle. Use Kaggle for compute and disk, and save the output CSV under `/kaggle/working` so it is included in the notebook output.

## Kaggle setup

1. Create a Kaggle Notebook.
2. Turn on Internet in notebook settings.
3. Clone this repo in the notebook:
   `https://github.com/vngclinh/MAI-verify-transcript.git`
4. Add secrets:
   - `HF_TOKEN`
   - `AZURE_SPEECH_ENDPOINT`
   - `AZURE_SPEECH_KEY`

If you already have a partial output CSV, upload it too and copy it to `/kaggle/working/outputs/` before running.

## Notebook cells

Clone repo:

```python
REPO_URL = "https://github.com/vngclinh/MAI-verify-transcript.git"
WORKDIR = "/kaggle/working/MAI-verify-transcript"
!rm -rf "$WORKDIR"
!git clone "$REPO_URL" "$WORKDIR"
```

Install dependencies:

```python
!pip install -q -r "$WORKDIR/requirements.txt"
```

Load secrets into environment variables:

```python
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
os.environ["AZURE_SPEECH_ENDPOINT"] = secrets.get_secret("AZURE_SPEECH_ENDPOINT")
os.environ["AZURE_SPEECH_KEY"] = secrets.get_secret("AZURE_SPEECH_KEY")
```

If resuming from a previous CSV uploaded as a Kaggle input dataset, copy it first:

```python
!mkdir -p /kaggle/working/MAI-verify-transcript/outputs
# Example only. Change this path to your uploaded previous output if present.
# !cp /kaggle/input/mai-output/viet_casual_speech_code_switching_mai.csv /kaggle/working/MAI-verify-transcript/outputs/
```

Run the automatic 5-parquet loop:

```python
%cd /kaggle/working/MAI-verify-transcript
!python scripts/verify_auto_parquet_batches.py \
  --output /kaggle/working/MAI-verify-transcript/outputs/viet_casual_speech_code_switching_mai.csv \
  --datasets-dir /kaggle/working/MAI-verify-transcript/datasets \
  --batch-size 5 \
  --save-every 25
```

The script:

- Downloads 5 pending parquet files when no useful local parquet remains.
- Processes parquet files one at a time.
- Reads parquet row groups instead of loading the whole parquet into RAM.
- Saves progress every `--save-every` audio files.
- Deletes each parquet after it completes, unless `--keep-parquet` is set.
- Resumes from `/kaggle/working/MAI-verify-VietCasual/outputs/viet_casual_speech_code_switching_mai.csv`.

## Important notes

- Kaggle sessions have runtime limits. If the notebook stops, commit/save the output CSV, then use that CSV as input for the next run.
- `/kaggle/working` is included in notebook output after commit. It is not permanent until you save/commit.
- Google Drive mounting is native to Colab, not Kaggle. On Kaggle, the usual workflow is to save to `/kaggle/working`, commit, then download the output or attach it as an input dataset for the next run.
- Keep the output CSV closed while running locally. On Kaggle this lock issue usually does not happen.
