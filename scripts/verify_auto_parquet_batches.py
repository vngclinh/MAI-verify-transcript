import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

from verify_local_audio_batch import OUTPUT_EXTRA_COLUMNS, is_checked
from verify_viet_casual_mai import (
    DATASET_REPOS,
    download_parquet,
    extract_transcript,
    score_row,
    transcribe_audio,
)


def autosave_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + ".autosave" + output_path.suffix)


def checked_count(df: pd.DataFrame) -> int:
    if "mai_checked" not in df.columns:
        return 0
    return int(df["mai_checked"].map(is_checked).sum())


def read_csv_with_extra_columns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in OUTPUT_EXTRA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def ensure_output_frame(input_path: Path, output_path: Path) -> pd.DataFrame:
    candidates = [path for path in [output_path, autosave_path(output_path)] if path.exists()]
    if not candidates:
        return read_csv_with_extra_columns(input_path)

    frames = [(path, read_csv_with_extra_columns(path)) for path in candidates]
    path, df = max(frames, key=lambda item: checked_count(item[1]))
    if path != output_path:
        print(f"Resuming from autosave checkpoint: {path}")
    return df


def save_output(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    fallback_path = autosave_path(output_path)
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    try:
        tmp_path.replace(output_path)
        return output_path
    except PermissionError:
        tmp_path.replace(fallback_path)
        print(
            f"Output is locked, saved checkpoint to {fallback_path}. "
            "Close the CSV viewer before the next final save."
        )
        return fallback_path


def group_local_path(datasets_dir: Path, dataset: str, parquet_file: str) -> Path:
    repo_id = DATASET_REPOS[dataset]
    return datasets_dir / repo_id / "data" / parquet_file


def pending_groups(df: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    mask = pd.Series(True, index=df.index)
    if args.dataset:
        mask &= df["dataset"].astype(str) == args.dataset
    if args.split:
        mask &= df["split"].astype(str) == args.split
    if not args.retry_checked:
        mask &= ~df["mai_checked"].map(is_checked)

    pending = df[mask].copy()
    groups = []
    for (dataset, split, parquet_file), group in pending.groupby(
        ["dataset", "split", "parquet_file"],
        sort=False,
    ):
        if dataset not in DATASET_REPOS:
            continue
        groups.append(
            {
                "dataset": dataset,
                "split": split,
                "parquet_file": parquet_file,
                "first_row": int(group.index.min()),
                "rows": len(group),
            }
        )
    groups.sort(key=lambda item: item["first_row"])
    return groups


def pick_existing_batch(groups: list[dict], datasets_dir: Path, batch_size: int) -> list[dict]:
    batch = []
    for group in groups:
        path = group_local_path(datasets_dir, group["dataset"], group["parquet_file"])
        if path.exists():
            group = dict(group)
            group["path"] = path
            group["source"] = "local"
            batch.append(group)
            if len(batch) >= batch_size:
                break
    return batch


def download_batch(groups: list[dict], args: argparse.Namespace, hf_token: str | None) -> list[dict]:
    batch = []
    for group in groups[: args.batch_size]:
        repo_id = DATASET_REPOS[group["dataset"]]
        parquet_path = download_parquet(
            repo_id=repo_id,
            filename=f"data/{group['parquet_file']}",
            datasets_dir=args.datasets_dir,
            hf_token=hf_token,
            backend=args.download_backend,
        )
        group = dict(group)
        group["path"] = Path(parquet_path)
        group["source"] = "downloaded"
        batch.append(group)
    return batch


def update_rows_for_file(
    df: pd.DataFrame,
    row_indexes: list[int],
    transcript: str,
    duration_ms: object,
    locale: object,
    confidence: object,
    latency: float,
    source: dict,
    checked_at: str,
) -> None:
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
        df.at[idx, "mai_audio_path"] = source["path"]
        df.at[idx, "mai_checked_at"] = checked_at


def process_one_parquet(
    df: pd.DataFrame,
    group: dict,
    args: argparse.Namespace,
    endpoint: str,
    key: str,
    output_path: Path,
) -> tuple[int, bool]:
    parquet_path = Path(group["path"])
    row_mask = (
        (df["dataset"].astype(str) == group["dataset"])
        & (df["split"].astype(str) == group["split"])
        & (df["parquet_file"].astype(str) == group["parquet_file"])
    )
    if not args.retry_checked:
        row_mask &= ~df["mai_checked"].map(is_checked)

    target_names = set(df.loc[row_mask, "file_name"].astype(str))
    had_error = False
    processed = 0
    processed_names: set[str] = set()

    print(
        f"Processing {group['dataset']}/{group['split']}/{group['parquet_file']} "
        f"({len(target_names):,} pending CSV rows)"
    )
    parquet_file = pq.ParquetFile(parquet_path)
    with tqdm(total=len(target_names), desc=group["parquet_file"]) as progress:
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_idx,
                columns=["audio", "duration"],
            )
            row_group = table.to_pandas()
            for row in row_group.itertuples(index=False):
                audio = row.audio
                file_name = Path(audio["path"]).name
                if file_name not in target_names or file_name in processed_names:
                    continue

                file_mask = row_mask & (df["file_name"].astype(str) == file_name)
                row_indexes = list(df.index[file_mask])
                if not row_indexes:
                    continue

                source = {
                    "bytes": audio["bytes"],
                    "path": audio["path"],
                    "duration": row.duration,
                }
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
                    update_rows_for_file(
                        df=df,
                        row_indexes=row_indexes,
                        transcript=transcript,
                        duration_ms=duration_ms,
                        locale=locale,
                        confidence=confidence,
                        latency=latency,
                        source=source,
                        checked_at=checked_at,
                    )
                except Exception as exc:  # noqa: BLE001 - keep long runs resumable.
                    had_error = True
                    for idx in row_indexes:
                        df.at[idx, "mai_checked"] = False
                        df.at[idx, "mai_status"] = "error"
                        df.at[idx, "mai_error"] = str(exc)[:1000]
                        df.at[idx, "mai_audio_path"] = source["path"]
                        df.at[idx, "mai_checked_at"] = checked_at

                processed += 1
                processed_names.add(file_name)
                progress.update(1)
                if processed % args.save_every == 0:
                    save_output(df, output_path)
                if args.sleep:
                    time.sleep(args.sleep)
            del row_group

    save_output(df, output_path)
    return processed, had_error


def delete_parquet(path: Path, args: argparse.Namespace, had_error: bool) -> None:
    if args.keep_parquet:
        return
    if had_error and not args.delete_even_if_errors:
        print(f"Keeping parquet because this shard had errors: {path}")
        return
    try:
        path.unlink()
        print(f"Deleted parquet: {path}")
    except FileNotFoundError:
        pass


def run(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file)
    hf_token = os.getenv("HF_TOKEN")
    endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")
    key = os.getenv("AZURE_SPEECH_KEY")
    if not endpoint or not key:
        raise SystemExit("Missing AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_KEY")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = ensure_output_frame(Path(args.input), output_path)
    datasets_dir = Path(args.datasets_dir)

    total_processed = 0
    batches_completed = 0
    while True:
        groups = pending_groups(df, args)
        if not groups:
            break

        batch = pick_existing_batch(groups, datasets_dir, args.batch_size)
        if batch:
            print(f"Found {len(batch)} already-downloaded parquet file(s) to process.")
        else:
            print(f"No pending local parquet found. Downloading next {args.batch_size} parquet file(s).")
            batch = download_batch(groups, args, hf_token)

        for group in batch:
            parquet_path = Path(group["path"])
            completed = False
            had_error = False
            try:
                processed, had_error = process_one_parquet(
                    df=df,
                    group=group,
                    args=args,
                    endpoint=endpoint,
                    key=key,
                    output_path=output_path,
                )
                total_processed += processed
                completed = True
            finally:
                if completed:
                    delete_parquet(parquet_path, args, had_error)

        checked_total = checked_count(df)
        batches_completed += 1
        print(f"Batch complete. Total files processed this run: {total_processed:,}")
        print(f"Checked rows in output: {checked_total:,}")

        if args.max_parquet_batches and batches_completed >= args.max_parquet_batches:
            print("Stopped because --max-parquet-batches was reached.")
            break

    save_output(df, output_path)
    print(f"Done. Files processed this run: {total_processed:,}")
    print(f"Output CSV: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="viet_casual_speech_code_switching.csv")
    parser.add_argument("--output", default="outputs/viet_casual_speech_code_switching_mai.csv")
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--download-backend", choices=["requests", "hf"], default="requests")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--locale", default="vi")
    parser.add_argument("--transcribe-style", choices=["verbatim", "clean"], default="verbatim")
    parser.add_argument("--retry-checked", action="store_true")
    parser.add_argument("--keep-parquet", action="store_true")
    parser.add_argument("--delete-even-if-errors", action="store_true")
    parser.add_argument(
        "--max-parquet-batches",
        type=int,
        help="Optional safety stop after this many 5-parquet batches.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        raise SystemExit("\nStopped by user. Re-run the same command to resume.")
