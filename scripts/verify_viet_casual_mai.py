import argparse
import csv
import json
import os
import re
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, hf_hub_url
from jiwer import cer, wer
from rapidfuzz import fuzz
from tqdm import tqdm


DATASET_REPOS = {
    "VietCasualSpeechA": "NghiMe/VietCasualSpeechA",
    "VietCasualSpeechB": "NghiMe/VietCasualSpeechB",
}

OUTPUT_COLUMNS = [
    "row_id",
    "dataset",
    "split",
    "parquet_file",
    "file_name",
    "original_text",
    "cs_words",
    "num_cs_words",
    "dataset_duration_s",
    "mai_transcript",
    "mai_duration_ms",
    "mai_locale",
    "mai_confidence",
    "wer",
    "cer",
    "similarity",
    "cs_words_found",
    "cs_words_missing",
    "exact_normalized_match",
    "status",
    "error",
    "api_latency_s",
    "transcribed_at",
]


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_keyword(text: object) -> str:
    text = normalize_text(text)
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def parse_cs_words(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [w.strip() for w in str(value).split(",") if w.strip()]


def extract_transcript(payload: dict) -> tuple[str, object, object, object]:
    phrases = payload.get("phrases") or []
    combined = payload.get("combinedPhrases") or []
    transcript = ""
    if combined:
        transcript = " ".join(p.get("text", "") for p in combined if p.get("text"))
    elif phrases:
        transcript = " ".join(p.get("text", "") for p in phrases if p.get("text"))
    locale = phrases[0].get("locale") if phrases else None
    confidence = phrases[0].get("confidence") if phrases else None
    return transcript.strip(), payload.get("durationMilliseconds"), locale, confidence


def score_row(reference: str, hypothesis: str, cs_words: object) -> dict:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    words = parse_cs_words(cs_words)
    found = []
    missing = []
    hyp_keyword = normalize_for_keyword(hypothesis)
    hyp_for_contains = f" {hyp_keyword} "
    for word in words:
        word_norm = normalize_for_keyword(word)
        if not word_norm:
            continue
        if f" {word_norm} " in hyp_for_contains or word_norm in hyp_keyword:
            found.append(word)
        else:
            missing.append(word)
    if ref_norm and hyp_norm:
        row_wer = wer(ref_norm, hyp_norm)
        row_cer = cer(ref_norm, hyp_norm)
        similarity = fuzz.ratio(ref_norm, hyp_norm) / 100.0
    else:
        row_wer = None
        row_cer = None
        similarity = None
    return {
        "wer": row_wer,
        "cer": row_cer,
        "similarity": similarity,
        "cs_words_found": ", ".join(found),
        "cs_words_missing": ", ".join(missing),
        "exact_normalized_match": ref_norm == hyp_norm,
    }


def transcribe_audio(
    endpoint: str,
    key: str,
    file_name: str,
    audio_bytes: bytes,
    timeout: int,
    retries: int,
    transcribe_style: str,
    force_locale: str | None,
) -> tuple[dict, float]:
    url = endpoint.rstrip("/") + "/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
    definition = {
        "enhancedMode": {
            "enabled": True,
            "model": "MAI-Transcribe-2",
            "modelOptions": {"transcribeStyle": transcribe_style},
        }
    }
    if force_locale:
        definition["locales"] = [force_locale]

    suffix = Path(file_name).suffix or ".wav"
    last_error = None
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        for attempt in range(retries + 1):
            started = time.perf_counter()
            try:
                with open(tmp_path, "rb") as fh:
                    response = requests.post(
                        url,
                        headers={"Ocp-Apim-Subscription-Key": key},
                        files={
                            "audio": (file_name, fh, "audio/wav"),
                            "definition": (
                                None,
                                json.dumps(definition, ensure_ascii=False),
                                "application/json",
                            ),
                        },
                        timeout=timeout,
                    )
                latency = time.perf_counter() - started
                if response.status_code == 200:
                    return response.json(), latency
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except Exception as exc:  # noqa: BLE001 - keep batch jobs resumable.
                latency = time.perf_counter() - started
                last_error = repr(exc)
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
        raise RuntimeError(last_error)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def load_done_keys(output_path: Path, retry_failed: bool) -> set[int]:
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=["row_id", "status"])
    except Exception:
        return set()
    if retry_failed:
        existing = existing[existing["status"] == "ok"]
    return set(existing["row_id"].astype(int).tolist())


def append_row(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def parquet_groups(df: pd.DataFrame) -> list[tuple[tuple[str, str, str], pd.DataFrame]]:
    grouped = []
    for key, group in df.groupby(["dataset", "split", "parquet_file"], sort=True):
        grouped.append((key, group.sort_values("row_id")))
    return grouped


def download_parquet(
    repo_id: str,
    filename: str,
    datasets_dir: str,
    hf_token: str | None,
    backend: str,
) -> str:
    local_path = Path(datasets_dir) / repo_id / filename
    if local_path.exists():
        return str(local_path)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id}/{filename} ...")

    if backend == "hf":
        return hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=hf_token,
            local_dir=str(Path(datasets_dir) / repo_id),
        )

    part_path = local_path.with_suffix(local_path.suffix + ".part")
    downloaded = part_path.stat().st_size if part_path.exists() else 0
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"

    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        if downloaded and response.status_code != 206:
            downloaded = 0
            part_path.unlink(missing_ok=True)
            headers.pop("Range", None)
            response.close()
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as retry:
                retry.raise_for_status()
                _stream_to_file(retry, part_path, local_path, 0)
                return str(local_path)
        response.raise_for_status()
        _stream_to_file(response, part_path, local_path, downloaded)
    return str(local_path)


def _stream_to_file(
    response: requests.Response,
    part_path: Path,
    local_path: Path,
    downloaded: int,
) -> None:
    content_length = int(response.headers.get("Content-Length") or 0)
    total = downloaded + content_length if content_length else None
    mode = "ab" if downloaded else "wb"
    with tqdm(
        total=total,
        initial=downloaded,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=local_path.name,
    ) as progress:
        with part_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
    part_path.replace(local_path)


def run(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file)
    hf_token = os.getenv("HF_TOKEN")
    speech_endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    if not speech_endpoint or not speech_key:
        raise SystemExit("Missing AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_KEY in .env")

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    df = pd.read_csv(input_path)
    df.insert(0, "row_id", range(len(df)))

    done = load_done_keys(output_path, retry_failed=args.retry_failed)
    todo = df[~df["row_id"].isin(done)].copy()
    if args.limit is not None:
        todo = todo.head(args.limit)

    total_to_process = len(todo)
    print(f"Rows in input: {len(df):,}")
    print(f"Rows already done: {len(done):,}")
    print(f"Rows to process this run: {total_to_process:,}")

    processed = 0
    for (dataset, _split, parquet_file), group in parquet_groups(todo):
        repo_id = DATASET_REPOS.get(dataset)
        if not repo_id:
            raise ValueError(f"Unknown dataset: {dataset}")
        filename = f"data/{parquet_file}"
        parquet_path = download_parquet(
            repo_id=repo_id,
            filename=filename,
            datasets_dir=args.datasets_dir,
            hf_token=hf_token,
            backend=args.download_backend,
        )
        parquet_path_obj = Path(parquet_path).resolve()
        delete_after_processing = (
            args.delete_parquet_after_processing
            and parquet_path_obj.is_relative_to(Path(args.datasets_dir).resolve())
        )
        shard_completed = False

        try:
            target_names = set(group["file_name"].astype(str))
            table = pq.read_table(parquet_path, columns=["audio", "text", "duration"])
            shard = table.to_pandas()
            shard["file_name"] = shard["audio"].map(lambda a: Path(a["path"]).name)
            shard = shard[shard["file_name"].isin(target_names)]
            audio_by_name = {row.file_name: row for row in shard.itertuples(index=False)}

            for csv_row in tqdm(
                group.itertuples(index=False),
                total=len(group),
                desc=f"{dataset}/{parquet_file}",
            ):
                source = audio_by_name.get(str(csv_row.file_name))
                base = {
                    "row_id": csv_row.row_id,
                    "dataset": csv_row.dataset,
                    "split": csv_row.split,
                    "parquet_file": csv_row.parquet_file,
                    "file_name": csv_row.file_name,
                    "original_text": csv_row.text,
                    "cs_words": csv_row.cs_words,
                    "num_cs_words": csv_row.num_cs_words,
                    "transcribed_at": datetime.now(timezone.utc).isoformat(),
                }
                if source is None:
                    append_row(
                        output_path,
                        {
                            **base,
                            "status": "missing_audio",
                            "error": "file_name not found in parquet shard",
                        },
                    )
                    continue

                try:
                    payload, latency = transcribe_audio(
                        endpoint=speech_endpoint,
                        key=speech_key,
                        file_name=csv_row.file_name,
                        audio_bytes=source.audio["bytes"],
                        timeout=args.timeout,
                        retries=args.retries,
                        transcribe_style=args.transcribe_style,
                        force_locale=args.locale,
                    )
                    transcript, mai_duration_ms, locale, confidence = extract_transcript(payload)
                    scores = score_row(csv_row.text, transcript, csv_row.cs_words)
                    append_row(
                        output_path,
                        {
                            **base,
                            **scores,
                            "dataset_duration_s": source.duration,
                            "mai_transcript": transcript,
                            "mai_duration_ms": mai_duration_ms,
                            "mai_locale": locale,
                            "mai_confidence": confidence,
                            "status": "ok",
                            "api_latency_s": round(latency, 3),
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - write failures and continue.
                    append_row(
                        output_path,
                        {
                            **base,
                            "dataset_duration_s": getattr(source, "duration", ""),
                            "status": "error",
                            "error": str(exc)[:1000],
                        },
                    )
                processed += 1
                if args.sleep:
                    time.sleep(args.sleep)
            shard_completed = True
        finally:
            if delete_after_processing and shard_completed:
                try:
                    parquet_path_obj.unlink()
                    print(f"Deleted processed shard: {parquet_path_obj}")
                except OSError as exc:
                    print(f"Could not delete processed shard {parquet_path_obj}: {exc}")

    write_report(output_path, report_path)
    print(f"Processed this run: {processed:,}")
    print(f"CSV: {output_path}")
    print(f"Report: {report_path}")


def write_report(output_path: Path, report_path: Path) -> None:
    if not output_path.exists():
        return
    df = pd.read_csv(output_path)
    ok = df[df["status"] == "ok"].copy()
    status_counts = Counter(df["status"].fillna(""))
    lines = [
        "# MAI verification report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Rows written: {len(df):,}",
        "",
        "## Status",
        "",
    ]
    for status, count in status_counts.most_common():
        lines.append(f"- {status}: {count:,}")

    if len(ok):
        numeric_cols = ["wer", "cer", "similarity", "api_latency_s"]
        for col in numeric_cols:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")
        missing_cs = ok["cs_words_missing"].fillna("").astype(str).str.strip() != ""
        exact = ok["exact_normalized_match"].astype(str).str.lower().isin(["true", "1"])
        lines += [
            "",
            "## Aggregate quality",
            "",
            f"- OK rows: {len(ok):,}",
            f"- Exact normalized matches: {int(exact.sum()):,} ({exact.mean():.2%})",
            f"- Rows with missing code-switch words: {int(missing_cs.sum()):,} ({missing_cs.mean():.2%})",
            f"- Mean WER: {ok['wer'].mean():.4f}",
            f"- Median WER: {ok['wer'].median():.4f}",
            f"- Mean CER: {ok['cer'].mean():.4f}",
            f"- Median similarity: {ok['similarity'].median():.4f}",
            f"- Median API latency: {ok['api_latency_s'].median():.2f}s",
            "",
            "## By dataset",
            "",
        ]
        by_dataset = ok.groupby("dataset").agg(
            rows=("row_id", "count"),
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            median_similarity=("similarity", "median"),
        )
        lines.append(by_dataset.to_markdown(floatfmt=".4f"))

        worst = ok.sort_values(["wer", "cer"], ascending=False).head(20)
        lines += [
            "",
            "## Highest WER examples",
            "",
        ]
        for row in worst.itertuples(index=False):
            lines += [
                f"### {row.dataset}/{row.split}/{row.file_name}",
                "",
                f"- WER: {float(row.wer):.4f}; CER: {float(row.cer):.4f}; similarity: {float(row.similarity):.4f}",
                f"- Missing code-switch words: {row.cs_words_missing if isinstance(row.cs_words_missing, str) and row.cs_words_missing else 'None'}",
                f"- Label: {row.original_text}",
                f"- MAI: {row.mai_transcript}",
                "",
            ]

        cs_counter = Counter()
        for value in ok["cs_words_missing"].dropna():
            for word in parse_cs_words(value):
                cs_counter[word.lower()] += 1
        if cs_counter:
            lines += ["## Most missed code-switch words", ""]
            for word, count in cs_counter.most_common(30):
                lines.append(f"- {word}: {count}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="viet_casual_speech_code_switching.csv")
    parser.add_argument("--output", default="outputs/mai_transcripts.csv")
    parser.add_argument("--report", default="outputs/mai_report.md")
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--download-backend", choices=["requests", "hf"], default="requests")
    parser.add_argument("--delete-parquet-after-processing", action="store_true")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--locale", default="vi")
    parser.add_argument("--transcribe-style", choices=["verbatim", "clean"], default="verbatim")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        raise SystemExit("\nStopped by user. Re-run the same command to resume.")
