#!/usr/bin/env python3
"""Stream a reproducible 65/30/5 tokenizer sample to the Rust trainer.

The script is deliberately a data adapter, not a tokenizer implementation.
It uses a Hugging Face iterable dataset for remote FineWeb and PyArrow batches
for local Ficbook Parquet, then writes a compact binary protocol to stdout:

    b"BDHCQDS1"                          # protocol magic
    u8 source_id + u64_le length + UTF-8 # repeated
    0xff + u64_le(0)                     # explicit successful end

Source ID 0xfe carries a UTF-8 error message. Explicit terminal frames avoid
depending on pipe EOF, which a dataset library's helper process may keep open.

All status messages go to stderr so they cannot corrupt the corpus stream.
No rating, tag, profanity or semantic-content filter is applied.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import random
import struct
import sys
from pathlib import Path
from typing import Iterable, Iterator


MAGIC = b"BDHCQDS1"
SOURCE_NAMES = ("fineweb2_hq", "ficbook", "ru_classic")
MAX_DOCUMENT_BYTES = 1 * 1024 * 1024
REPORT_EVERY_BYTES = 64 * 1024 * 1024
STREAM_ERROR = 254
STREAM_END = 255


def split_budget(total: int) -> tuple[int, int, int]:
    """Use integer arithmetic and assign rounding remainder to ru-classic."""
    fineweb = total * 65 // 100
    ficbook = total * 30 // 100
    return fineweb, ficbook, total - fineweb - ficbook


class FrameWriter:
    """Write valid UTF-8 frames while enforcing per-source byte quotas."""

    def __init__(self, output, budgets: tuple[int, int, int]) -> None:
        self.output = output
        self.budgets = budgets
        self.bytes = [0, 0, 0]
        self.sequences = [0, 0, 0]
        self.next_report = [REPORT_EVERY_BYTES] * 3
        output.write(MAGIC)
        output.flush()

    def done(self, source: int) -> bool:
        return self.bytes[source] >= self.budgets[source]

    def emit(self, source: int, text: str | None) -> None:
        """Split one document into bounded frames without splitting UTF-8."""
        if not text or self.done(source):
            return
        data = text.encode("utf-8")
        offset = 0
        while offset < len(data) and not self.done(source):
            remaining = self.budgets[source] - self.bytes[source]
            wanted = min(MAX_DOCUMENT_BYTES, remaining, len(data) - offset)
            end = offset + wanted

            # If `end` points into a multibyte character, move to its start.
            while end > offset and end < len(data) and data[end] & 0b1100_0000 == 0b1000_0000:
                end -= 1

            # A final quota of 1--3 bytes may be smaller than the next codepoint.
            # Include that one codepoint; source totals can exceed plan by at
            # most three bytes, which is explicit in the generated manifest.
            if end == offset:
                end = min(offset + wanted, len(data))
                while end < len(data) and data[end] & 0b1100_0000 == 0b1000_0000:
                    end += 1

            chunk = data[offset:end]
            # This assertion also catches mistakes in boundary arithmetic.
            chunk.decode("utf-8")
            self.output.write(bytes((source,)))
            self.output.write(struct.pack("<Q", len(chunk)))
            self.output.write(chunk)
            self.bytes[source] += len(chunk)
            self.sequences[source] += 1
            offset = end

            if self.bytes[source] >= self.next_report[source]:
                print(
                    f"[{SOURCE_NAMES[source]}] {self.bytes[source]:,} / "
                    f"{self.budgets[source]:,} UTF-8 bytes",
                    file=sys.stderr,
                )
                self.next_report[source] += REPORT_EVERY_BYTES

    def finish_source(self, source: int) -> None:
        if not self.done(source):
            raise RuntimeError(
                f"{SOURCE_NAMES[source]} ended after {self.bytes[source]:,} bytes; "
                f"the requested quota is {self.budgets[source]:,}"
            )
        print(
            f"[{SOURCE_NAMES[source]}] complete: {self.sequences[source]:,} sequences, "
            f"{self.bytes[source]:,} UTF-8 bytes",
            file=sys.stderr,
        )

    def finish_stream(self) -> None:
        """End explicitly; EOF may be held open by a dataset worker process."""
        self.output.write(bytes((STREAM_END,)))
        self.output.write(struct.pack("<Q", 0))
        self.output.flush()

    def abort(self, error: Exception) -> None:
        """Send a bounded diagnostic to Rust before returning a failure code."""
        payload = str(error).encode("utf-8")[:MAX_DOCUMENT_BYTES]
        self.output.write(bytes((STREAM_ERROR,)))
        self.output.write(struct.pack("<Q", len(payload)))
        self.output.write(payload)
        self.output.flush()


def load_hf_datasets():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face datasets is missing. Install it with: "
            "python3 -m pip install -r scripts/requirements-tokenizer.txt"
        ) from error
    return load_dataset


def stream_fineweb(args: argparse.Namespace, writer: FrameWriter) -> None:
    """Project only `text`; do not transfer the dataset's large embeddings."""
    load_dataset = load_hf_datasets()
    dataset = load_dataset(
        args.fineweb_dataset,
        args.fineweb_config,
        split="train",
        streaming=True,
        columns=["text"],
        revision=args.fineweb_revision,
    )
    # A 10k shuffle buffer is useful for the real sample but wasteful for a
    # 650KB smoke test because the buffer is filled before the first yield.
    adaptive_buffer = min(args.shuffle_buffer, max(128, writer.budgets[0] // 4096))
    dataset = dataset.shuffle(seed=args.seed, buffer_size=adaptive_buffer)
    for row in dataset:
        writer.emit(0, row.get("text"))
        if writer.done(0):
            break
    writer.finish_source(0)


def ficbook_documents(
    row: dict, part_field: str, include_metadata: bool = False
) -> Iterator[str]:
    """Turn one Ficbook story into one sequence per chapter/part.

    By default only the part body is emitted. Optional metadata is not used by
    tokenizer or LM production jobs. No story is screened by its rating/tags;
    `clean_text` removes acquisition markup but is not a moderation filter.
    """
    metadata: list[str] = []
    if include_metadata:
        title = row.get("title")
        description = row.get("description")
        tags = row.get("tags") or []
        rating = row.get("rating")
        if title:
            metadata.append(str(title))
        if description:
            metadata.append(str(description))
        if tags:
            metadata.append("Теги: " + ", ".join(map(str, tags)))
        if rating:
            metadata.append("Рейтинг: " + str(rating))

    first = True
    for part in row.get("parts") or []:
        if not isinstance(part, dict):
            continue
        body = part.get(part_field)
        if not body:
            continue
        fields: list[str] = []
        if first:
            fields.extend(metadata)
            first = False
        if include_metadata:
            part_title = part.get("title")
            if part_title:
                fields.append(str(part_title))
        fields.append(str(body))
        yield "\n\n".join(fields)


def stream_ficbook(args: argparse.Namespace, writer: FrameWriter) -> None:
    # `datasets` 4.8.5 + PyArrow 23's Dataset scanner reports corrupt Snappy
    # data for these valid nested Parquet files. ParquetFile.iter_batches reads
    # the same pages correctly and still projects only the five needed columns.
    import pyarrow.parquet as parquet

    files = sorted(glob.glob(args.ficbook_glob))
    if not files:
        raise RuntimeError(f"no Ficbook Parquet files match {args.ficbook_glob!r}")
    random.Random(args.seed + 1).shuffle(files)
    columns = (
        ["title", "description", "tags", "rating", "parts"]
        if args.ficbook_include_metadata
        else ["parts"]
    )
    for path in files:
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                for document in ficbook_documents(
                    row,
                    args.ficbook_part_field,
                    include_metadata=args.ficbook_include_metadata,
                ):
                    writer.emit(1, document)
                    if writer.done(1):
                        writer.finish_source(1)
                        return
    writer.finish_source(1)


def random_file_windows(path: Path, seed: int, window_bytes: int) -> Iterable[str]:
    """Read large UTF-8 text files from shuffled windows, not only the prefix."""
    size = path.stat().st_size
    offsets = list(range(0, size, window_bytes))
    random.Random(seed).shuffle(offsets)
    with path.open("rb") as source:
        for offset in offsets:
            source.seek(offset)
            if offset:
                source.readline()  # discard the partial first line
            data = source.read(window_bytes)
            # The final codepoint or line can cross a window. Dropping only that
            # boundary fragment avoids inserting replacement characters.
            yield data.decode("utf-8", errors="ignore")


def stream_classics(args: argparse.Namespace, writer: FrameWriter) -> None:
    path = Path(args.classic_file)
    if not path.is_file():
        raise RuntimeError(f"ru-classic text file does not exist: {path}")
    for document in random_file_windows(path, args.seed + 2, 4 * 1024 * 1024):
        writer.emit(2, document)
        if writer.done(2):
            break
    writer.finish_source(2)


def stream_smoke_fixture(writer: FrameWriter, selected_source: int | None = None) -> None:
    """Dependency-free corpus used by CI and local plumbing checks."""
    fixtures = (
        (
            "Русская веб-страница: новости, числа 2026 и URL https://example.org/.\n",
            "FineWeb также содержит English fragments, code() и emoji 🦔.\n",
        ),
        (
            "Это фикбук-текст: проза первой части без служебной метадаты.\n",
            "Авторская речь, диалог — и разнообразная разговорная лексика.\n",
        ),
        (
            "Александр Пушкин. Мороз и солнце; день чудесный!\n",
            "Классическая русская проза сохраняет ё, кавычки «ёлочки» и тире.\n",
        ),
    )
    sources = range(3) if selected_source is None else (selected_source,)
    for source in sources:
        for document in itertools.cycle(fixtures[source]):
            writer.emit(source, document)
            if writer.done(source):
                break
        writer.finish_source(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-bytes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fineweb-dataset", default="epfml/FineWeb2-HQ")
    parser.add_argument("--fineweb-config", default="rus_Cyrl")
    parser.add_argument(
        "--fineweb-revision",
        default="c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
    )
    parser.add_argument("--ficbook-glob", default="datasets/ficbook/*.parquet")
    parser.add_argument(
        "--ficbook-part-field", choices=("clean_text", "text"), default="clean_text"
    )
    parser.add_argument(
        "--ficbook-include-metadata",
        action="store_true",
        help="diagnostic opt-in for story/chapter metadata; production omits it",
    )
    parser.add_argument("--classic-file", default="datasets/ru-classic.txt")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--smoke-fixture", action="store_true")
    parser.add_argument(
        "--source",
        choices=SOURCE_NAMES,
        help="stream only one source (used by the exact-token pretraining packer)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_bytes <= 0:
        raise ValueError("--sample-bytes must be positive")
    selected_source = SOURCE_NAMES.index(args.source) if args.source else None
    budgets = (
        tuple(args.sample_bytes for _ in SOURCE_NAMES)
        if selected_source is not None
        else split_budget(args.sample_bytes)
    )
    writer = FrameWriter(sys.stdout.buffer, budgets)
    try:
        if args.smoke_fixture:
            stream_smoke_fixture(writer, selected_source)
        elif selected_source == 0:
            stream_fineweb(args, writer)
        elif selected_source == 1:
            stream_ficbook(args, writer)
        elif selected_source == 2:
            stream_classics(args, writer)
        else:
            # Sequential source reads are intentional. BPE accumulates counts,
            # so interleaving does not change its result, and only one remote
            # iterator needs to be live at a time.
            stream_fineweb(args, writer)
            stream_ficbook(args, writer)
            stream_classics(args, writer)
    except Exception as error:
        print(f"tokenizer sampler failed: {error}", file=sys.stderr)
        writer.abort(error)
        return 1
    writer.finish_stream()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BrokenPipeError:
        # Rust exited early (usually because BPE itself failed).
        exit_code = 1
    except Exception as error:
        print(f"tokenizer sampler failed before streaming: {error}", file=sys.stderr)
        exit_code = 1

    # Hugging Face/fsspec may leave a prefetch helper alive after an iterable is
    # abandoned at its byte quota. All protocol and diagnostic bytes have been
    # flushed by FrameWriter, so forcefully terminating this dedicated adapter
    # is safer than making Rust wait for an unrelated background retry thread.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)
