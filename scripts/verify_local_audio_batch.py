import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from verify_viet_casual_mai import extract_transcript, score_row, transcribe_audio


AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

OUTPUT_EXTRA_COLUMNS = [
    "mai_checked",
    "mai_transcript",
    "mai_status",
    "mai_error",
    "mai_duration_ms",
    "mai_locale",
    "mai_confidence",
    "mai_wer",
    "mai_cer",
    "mai_similarity",
    "mai_cs_words_found",
    "mai_cs_words_missing",
    "mai_exact_normalized_match",
    "mai_api_latency_s",
    "mai_audio_path",
    "mai_checked_at",
]


def find_audio_files(audio_dir: Path) -> dict[str, Path]:
    files_by_name: dict[str, Path] = {}
    duplicates = []
    for path in audio_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        key = path.name
        if key in files_by_name:
            duplicates.append(key)
            continue
        files_by_name[key] = path
    if duplicates:
        print(
            "Warning: duplicate audio basenames in the batch; using the first one for: "
            + ", ".join(sorted(set(duplicates))[:20])
        )
    return files_by_name


def ensure_output_frame(input_path: Path, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        df = pd.read_csv(output_path)
    else:
        df = pd.read_csv(input_path)
    for col in OUTPUT_EXTRA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if args.dataset:
        mask &= df["dataset"].astype(str) == args.dataset
    if args.split:
        mask &= df["split"].astype(str) == args.split
    if args.parquet_file:
        mask &= df["parquet_file"].astype(str) == args.parquet_file
    return mask


def is_checked(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def safe_delete_audio_dir(audio_dir: Path) -> None:
    target = audio_dir.resolve()
    cwd = Path.cwd().resolve()
    if target == cwd or target == cwd.parent or target.parent == target:
        raise RuntimeError(f"Refusing to delete unsafe audio directory: {target}")
    if not target.exists():
        return
    shutil.rmtree(target)
    print(f"Deleted audio batch directory: {target}")


def delete_files(paths: list[Path], audio_dir: Path) -> None:
    root = audio_dir.resolve()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"Refusing to delete file outside audio dir: {resolved}")
        try:
            resolved.unlink()
        except FileNotFoundError:
            pass
    for directory in sorted(
        {p.parent.resolve() for p in paths},
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        while directory != root and directory.exists():
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent
    print(f"Deleted matched audio files: {len(paths)}")


def run(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file)
    endpoint = args.azure_speech_endpoint
    key = args.azure_speech_key
    if not endpoint or not key:
        import os

        endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")
        key = os.getenv("AZURE_SPEECH_KEY")
    if not endpoint or not key:
        raise SystemExit("Missing AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_KEY")

    input_path = Path(args.input)
    output_path = Path(args.output)
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise SystemExit(f"Audio directory does not exist: {audio_dir}")

    df = ensure_output_frame(input_path, output_path)
    candidate_mask = apply_filters(df, args)
    files_by_name = find_audio_files(audio_dir)
    matched_names = sorted(set(df.loc[candidate_mask, "file_name"].astype(str)) & set(files_by_name))
    if args.limit is not None:
        matched_names = matched_names[: args.limit]

    print(f"Audio files in batch: {len(files_by_name):,}")
    print(f"Matched file names in CSV: {len(matched_names):,}")
    print(f"Output CSV: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    successful_files: list[Path] = []
    had_error = False

    for file_name in tqdm(matched_names, desc="MAI local batch"):
        row_mask = candidate_mask & (df["file_name"].astype(str) == file_name)
        if not args.retry_checked:
            row_mask &= ~df["mai_checked"].map(is_checked)
        row_indexes = list(df.index[row_mask])
        if not row_indexes:
            continue

        audio_path = files_by_name[file_name]
        checked_at = datetime.now(timezone.utc).isoformat()
        try:
            audio_bytes = audio_path.read_bytes()
            payload, latency = transcribe_audio(
                endpoint=endpoint,
                key=key,
                file_name=file_name,
                audio_bytes=audio_bytes,
                timeout=args.timeout,
                retries=args.retries,
                transcribe_style=args.transcribe_style,
                force_locale=args.locale,
            )
            transcript, duration_ms, locale, confidence = extract_transcript(payload)
            for idx in row_indexes:
                scores = score_row(df.at[idx, "text"], transcript, df.at[idx, "cs_words"])
                df.at[idx, "mai_checked"] = True
                df.at[idx, "mai_transcript"] = transcript
                df.at[idx, "mai_status"] = "ok"
                df.at[idx, "mai_error"] = ""
                df.at[idx, "mai_duration_ms"] = duration_ms
                df.at[idx, "mai_locale"] = locale
                df.at[idx, "mai_confidence"] = confidence
                df.at[idx, "mai_wer"] = scores["wer"]
                df.at[idx, "mai_cer"] = scores["cer"]
                df.at[idx, "mai_similarity"] = scores["similarity"]
                df.at[idx, "mai_cs_words_found"] = scores["cs_words_found"]
                df.at[idx, "mai_cs_words_missing"] = scores["cs_words_missing"]
                df.at[idx, "mai_exact_normalized_match"] = scores["exact_normalized_match"]
                df.at[idx, "mai_api_latency_s"] = round(latency, 3)
                df.at[idx, "mai_audio_path"] = str(audio_path)
                df.at[idx, "mai_checked_at"] = checked_at
            successful_files.append(audio_path)
        except Exception as exc:  # noqa: BLE001 - keep batch resumable.
            had_error = True
            for idx in row_indexes:
                df.at[idx, "mai_checked"] = False
                df.at[idx, "mai_status"] = "error"
                df.at[idx, "mai_error"] = str(exc)[:1000]
                df.at[idx, "mai_audio_path"] = str(audio_path)
                df.at[idx, "mai_checked_at"] = checked_at
        df.to_csv(output_path, index=False, encoding="utf-8")
        if args.sleep:
            time.sleep(args.sleep)

    df.to_csv(output_path, index=False, encoding="utf-8")
    if args.delete_matched_files_after_processing and successful_files:
        delete_files(successful_files, audio_dir)
    if args.delete_audio_dir_after_processing:
        if had_error and not args.delete_even_if_errors:
            print("Not deleting audio directory because at least one matched file had an error.")
        else:
            safe_delete_audio_dir(audio_dir)

    checked_total = df["mai_checked"].map(is_checked).sum()
    print(f"Checked rows in output: {checked_total:,}")
    print(f"Batch errors: {had_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--input", default="viet_casual_speech_code_switching.csv")
    parser.add_argument("--output", default="outputs/viet_casual_speech_code_switching_mai.csv")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--parquet-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--locale", default="vi")
    parser.add_argument("--transcribe-style", choices=["verbatim", "clean"], default="verbatim")
    parser.add_argument("--retry-checked", action="store_true")
    parser.add_argument("--delete-matched-files-after-processing", action="store_true")
    parser.add_argument("--delete-audio-dir-after-processing", action="store_true")
    parser.add_argument("--delete-even-if-errors", action="store_true")
    parser.add_argument("--azure-speech-endpoint")
    parser.add_argument("--azure-speech-key")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        raise SystemExit("\nStopped by user. Re-run the same command to resume.")
