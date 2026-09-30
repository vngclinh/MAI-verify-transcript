# MAI verification for VietCasualSpeech A/B

This project verifies the labels in `viet_casual_speech_code_switching.csv` by:

1. Downloading the required parquet shards from Hugging Face:
   - `NghiMe/VietCasualSpeechA`
   - `NghiMe/VietCasualSpeechB`
2. Extracting only the matching audio files listed in the CSV.
3. Sending each audio clip to Azure Speech Fast Transcription with `MAI-Transcribe-2`.
4. Writing an incremental transcript CSV and a Markdown report.

## Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The script reads `HF_TOKEN`, `AZURE_SPEECH_ENDPOINT`, and `AZURE_SPEECH_KEY` from `.env`.

## Smoke test

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\verify_viet_casual_mai.py --limit 3 --output outputs\mai_transcripts_smoke.csv --report outputs\mai_report_smoke.md
```

## Full run

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\verify_viet_casual_mai.py --output outputs\mai_transcripts.csv --report outputs\mai_report.md
```

The output CSV is append-only and resumable. If the run is interrupted, run the same command again and completed `ok` rows are skipped. Use `--retry-failed` to retry rows that previously ended with `error`.

Parquet downloads are also resumable. If you press `Ctrl+C` while a shard is downloading, a `.part` file is kept and the next run continues from it.

To keep disk usage low, process one parquet shard at a time and delete it after all matching audio in that shard has been transcribed:

```powershell
.\.venv\Scripts\python.exe scripts\verify_viet_casual_mai.py --output outputs\mai_transcripts.csv --report outputs\mai_report.md --delete-parquet-after-processing
```

Useful options:

- `--limit N`: process only the next N unprocessed rows.
- `--download-backend hf`: use Hugging Face's default downloader instead of the script's streaming downloader.
- `--delete-parquet-after-processing`: delete each parquet shard after its rows are done.
- `--sleep 0.2`: add delay between MAI calls if the service rate-limits.
- `--transcribe-style clean`: use MAI's clean transcript style instead of verbatim.
- `--locale vi`: force Vietnamese recognition. This is the default.

## Verify a local audio batch

If you already have a local folder of audio files, process only files whose basename appears in the CSV:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\verify_local_audio_batch.py --audio-dir D:\path\to\audio_batch --output outputs\viet_casual_speech_code_switching_mai.csv --delete-matched-files-after-processing
```

This writes a CSV with the original columns plus MAI columns such as `mai_checked`, `mai_transcript`, `mai_status`, `mai_wer`, and `mai_cer`.

Useful local-batch options:

- `--dataset VietCasualSpeechA`: only match rows from one dataset.
- `--split train`: only match one split.
- `--parquet-file train-00001-of-00103.parquet`: only match rows from one parquet shard.
- `--delete-matched-files-after-processing`: delete only matched audio files that were successfully checked.
- `--delete-audio-dir-after-processing`: delete the whole audio batch directory after processing. Use this only if the folder contains disposable batch data.

## Verify one local parquet batch

If the batch is a single parquet file, process only audio entries inside that parquet whose `audio.path` basename appears in the CSV:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\verify_parquet_batch.py --parquet D:\path\to\train-00001-of-00103.parquet --output outputs\viet_casual_speech_code_switching_mai.csv --delete-parquet-after-processing
```

By default this also requires the CSV column `parquet_file` to equal the parquet basename, which avoids accidental matches when the same `file_name` appears elsewhere. Add filters if needed:

```powershell
.\.venv\Scripts\python.exe scripts\verify_parquet_batch.py --parquet D:\path\to\train-00001-of-00103.parquet --dataset VietCasualSpeechA --split train --output outputs\viet_casual_speech_code_switching_mai.csv --delete-parquet-after-processing
```

Use `--match-all-parquet-names` only if the parquet filename does not match the CSV `parquet_file` values.

## Fully automatic 5-parquet loop

To process already-downloaded parquet files in groups of 5, delete them after each shard is done, then automatically download the next 5 pending parquet files and continue:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\verify_auto_parquet_batches.py --output outputs\viet_casual_speech_code_switching_mai.csv --batch-size 5
```

Behavior:

- First uses pending parquet files already present under `datasets\NghiMe\...\data`.
- Processes up to 5 parquet files per loop.
- Saves after every audio file, so reruns resume from `mai_checked = True`.
- Deletes each parquet after it completes successfully.
- When no pending local parquet remains, downloads the next 5 pending parquet files from Hugging Face and repeats.

Add `--dataset VietCasualSpeechA` or `--split train` if you want to limit the automatic loop.
Add `--keep-parquet` if you want to test without deleting parquet files.
