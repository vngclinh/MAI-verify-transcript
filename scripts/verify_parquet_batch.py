import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

from verify_local_audio_batch import OUTPUT_EXTRA_COLUMNS, is_checked
from verify_viet_casual_mai import extract_transcript, score_row, transcribe_audio


def ensure_output_frame(input_path: Path, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        df = pd.read_csv(output_path)
    else:
        df = pd.read_csv(input_path)
    for col in OUTPUT_EXTRA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def build_candidate_mask(
    df: pd.DataFrame,
    parquet_path: Path,
    args: argparse.Namespace,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if not args.match_all_parquet_names:
        mask &= df["parquet_file"].astype(str) == parquet_path.name
    if args.dataset:
        mask &= df["dataset"].astype(str) == args.dataset
    if args.split:
        mask &= df["split"].astype(str) == args.split
    if args.parquet_file:
        mask &= df["parquet_file"].astype(str) == args.parquet_file
    return mask


def load_audio_from_parquet(parquet_path: Path) -> dict[str, dict]:
    table = pq.read_table(parquet_path, columns=["audio", "duration"])
    shard = table.to_pandas()
    audio_by_name: dict[str, dict] = {}
    duplicates = []
    for row in shard.itertuples(index=False):
        audio = row.audio
        file_name = Path(audio["path"]).name
        if file_name in audio_by_name:
            duplicates.append(file_name)
            continue
        audio_by_name[file_name] = {
            "bytes": audio["bytes"],
            "path": audio["path"],
            "duration": row.duration,
        }
    if duplicates:
        print(
            "Warning: duplicate audio basenames in parquet; using the first one for: "
            + ", ".join(sorted(set(duplicates))[:20])
        )
    return audio_by_name


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

    parquet_path = Path(args.parquet).resolve()
    if not parquet_path.exists() or not parquet_path.is_file():
        raise SystemExit(f"Parquet file does not exist: {parquet_path}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = ensure_output_frame(input_path, output_path)
    candidate_mask = build_candidate_mask(df, parquet_path, args)

    print(f"Reading parquet: {parquet_path}")
    audio_by_name = load_audio_from_parquet(parquet_path)
    matched_names = sorted(set(df.loc[candidate_mask, "file_name"].astype(str)) & set(audio_by_name))
    if args.limit is not None:
        matched_names = matched_names[: args.limit]

    print(f"Audio rows in parquet: {len(audio_by_name):,}")
    print(f"Matched file names in CSV: {len(matched_names):,}")
    print(f"Output CSV: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    had_error = False
    processed_files = 0

    for file_name in tqdm(matched_names, desc=parquet_path.name):
        row_mask = candidate_mask & (df["file_name"].astype(str) == file_name)
        if not args.retry_checked:
            row_mask &= ~df["mai_checked"].map(is_checked)
        row_indexes = list(df.index[row_mask])
        if not row_indexes:
            continue

        source = audio_by_name[file_name]
        checked_at = datetime.now(timezone.utc).isoformat()
        try:
            payload, latency = transcribe_audio(
                endpoint=endpoint,
                key=key,
                file_name=file_name,
                audio_bytes=source["bytes"],
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
                df.at[idx, "mai_exact_normalized_match"] = scores[
                    "exact_normalized_match"
                ]
                df.at[idx, "mai_api_latency_s"] = round(latency, 3)
                df.at[idx, "mai_audio_path"] = source["path"]
                df.at[idx, "mai_checked_at"] = checked_at
        except Exception as exc:  # noqa: BLE001 - keep batch resumable.
            had_error = True
            for idx in row_indexes:
                df.at[idx, "mai_checked"] = False
                df.at[idx, "mai_status"] = "error"
                df.at[idx, "mai_error"] = str(exc)[:1000]
                df.at[idx, "mai_audio_path"] = source["path"]
                df.at[idx, "mai_checked_at"] = checked_at
        processed_files += 1
        df.to_csv(output_path, index=False, encoding="utf-8")
        if args.sleep:
            time.sleep(args.sleep)

    df.to_csv(output_path, index=False, encoding="utf-8")

    if args.delete_parquet_after_processing:
        if had_error and not args.delete_even_if_errors:
            print("Not deleting parquet because at least one matched file had an error.")
        else:
            parquet_path.unlink()
            print(f"Deleted parquet: {parquet_path}")

    checked_total = df["mai_checked"].map(is_checked).sum()
    print(f"Processed files from this parquet: {processed_files:,}")
    print(f"Checked rows in output: {checked_total:,}")
    print(f"Batch errors: {had_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--input", default="viet_casual_speech_code_switching.csv")
    parser.add_argument("--output", default="outputs/viet_casual_speech_code_switching_mai.csv")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--parquet-file")
    parser.add_argument("--match-all-parquet-names", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--locale", default="vi")
    parser.add_argument("--transcribe-style", choices=["verbatim", "clean"], default="verbatim")
    parser.add_argument("--retry-checked", action="store_true")
    parser.add_argument("--delete-parquet-after-processing", action="store_true")
    parser.add_argument("--delete-even-if-errors", action="store_true")
    parser.add_argument("--azure-speech-endpoint")
    parser.add_argument("--azure-speech-key")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        raise SystemExit("\nStopped by user. Re-run the same command to resume.")
