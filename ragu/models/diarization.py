"""
Speaker diarization: who spoke when, and how that maps onto recognized text.

Diarization is a sibling of :mod:`ragu.models.asr`, not a part of it. The two
read the same recording independently and answer different questions - one
"what was said", the other "who was speaking" - and marrying their answers is
a third thing, :func:`assign_speakers`, which is a function rather than a
model because it involves no inference at all.

Keeping them apart also lets a caller skip diarization entirely when the ASR
backend already returns speaker labels, which some hosted models do.

Example::

    from ragu.models.asr import ASRTransformers
    from ragu.models.diarization import DiarizationPyannote, assign_speakers

    asr = ASRTransformers(return_timestamps="word")
    diarization = DiarizationPyannote()

    transcript = await asr.transcribe("meeting.wav")
    turns = await diarization.diarize("meeting.wav")
    transcript = assign_speakers(transcript, turns)
"""
import asyncio
import inspect
import os
import tempfile
from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Sequence

from ragu.utils.batching import run_batched
from ragu.common.logger import logger
from ragu.models.asr import (
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    join_word_texts,
)
from ragu.utils.ragu_utils import experimental

DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# Environment variables searched for a HuggingFace token
HF_TOKEN_VARIABLES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")

DEFAULT_BATCH_SIZE = 4


@dataclass(slots=True)
class SpeakerTurn:
    """
    A stretch of audio attributed to one speaker.

    :param start: Start offset in seconds from the beginning of the recording.
    :param end: End offset in seconds.
    :param speaker: Speaker label, such as ``"SPEAKER_00"``.
    """

    start: float
    end: float
    speaker: str

    def shifted(self, offset: float) -> "SpeakerTurn":
        """
        Return a copy moved along the timeline.

        :param offset: Seconds to add to both bounds.
        :type offset: float
        :returns: Shifted copy.
        :rtype: SpeakerTurn
        """
        return SpeakerTurn(
            start=self.start + offset,
            end=self.end + offset,
            speaker=self.speaker,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the turn for storage in document metadata.

        :returns: JSON-serializable mapping.
        :rtype: dict[str, Any]
        """
        return {"start": self.start, "end": self.end, "speaker": self.speaker}


def _token_kwargs(pipeline_class: Any, token: str) -> dict[str, Any]:
    """
    Name the authentication argument the way this pyannote version spells it.

    :param pipeline_class: The ``pyannote.audio.Pipeline`` class.
    :type pipeline_class: Any
    :param token: HuggingFace token.
    :type token: str
    :returns: Keyword arguments for ``from_pretrained``.
    :rtype: dict[str, Any]
    """
    try:
        parameters = inspect.signature(pipeline_class.from_pretrained).parameters
    except (TypeError, ValueError):
        # Signature unavailable (C extension, heavy decoration): assume 4.x,
        # which is what a fresh install gets.
        return {"token": token}

    if "token" in parameters:
        return {"token": token}
    if "use_auth_token" in parameters:
        return {"use_auth_token": token}

    # Neither name is declared: the callable takes **kwargs, so pass the
    # current spelling and let it decide.
    return {"token": token}


def parse_pyannote_output(output: Any) -> List[SpeakerTurn]:
    """
    Read speaker turns out of whatever a pyannote pipeline returned.

    :param output: Whatever the pipeline call returned.
    :type output: Any
    :returns: Speaker turns, in the order the pipeline reported them.
    :rtype: List[SpeakerTurn]
    :raises RuntimeError: If no speaker turns can be read from the object.
    """
    annotation = getattr(output, "speaker_diarization", output)

    itertracks = getattr(annotation, "itertracks", None)
    if callable(itertracks):
        return [
            SpeakerTurn(start=float(segment.start), end=float(segment.end), speaker=str(label))
            for segment, _track, label in itertracks(yield_label=True)
        ]

    try:
        entries = list(annotation)
    except TypeError as exc:
        raise RuntimeError(
            f"pyannote returned {type(output).__name__}, which is neither an Annotation "
            f"nor iterable; this adapter supports pyannote.audio 3.x and 4.x"
        ) from exc

    turns: list[SpeakerTurn] = []
    for entry in entries:
        # 4.x yields (segment, label); a bare Annotation yields segments only.
        if not isinstance(entry, tuple):
            raise RuntimeError(
                f"pyannote returned {type(output).__name__} yielding "
                f"{type(entry).__name__} without speaker labels"
            )

        segment, label = entry[0], entry[-1]
        turns.append(
            SpeakerTurn(start=float(segment.start), end=float(segment.end), speaker=str(label))
        )

    return turns


class Diarization(ABC):
    """
    Abstract interface for speaker diarization backends.

    Implementations answer "who spoke when" for one recording. Video is not
    handled here: callers extract the audio track first (see
    :func:`ragu.utils.audio.extract_audio`).
    """

    @abstractmethod
    async def diarize(
        self,
        audio: str | Path | bytes,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> List[SpeakerTurn]:
        """
        Find the speaker turns of a recording.

        :param audio: Path to an audio file, or its encoded bytes.
        :type audio: str | Path | bytes
        :param num_speakers: Exact number of speakers, when it is known.
        :type num_speakers: int | None
        :param min_speakers: Lower bound on the speaker count.
        :type min_speakers: int | None
        :param max_speakers: Upper bound on the speaker count.
        :type max_speakers: int | None
        :returns: Speaker turns in timeline order.
        :rtype: List[SpeakerTurn]
        """

    async def diarize_batch(
        self,
        audios: Sequence[str | Path | bytes],
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        desc: str | None = "Diarizing",
        continue_on_error: bool = False,
    ) -> List[List[SpeakerTurn] | None]:
        """
        Diarize several recordings, `batch_size` at a time.

        The speaker bounds apply to every recording, so pass them only for a
        set that genuinely shares them.

        :param audios: Audio paths or encoded bytes.
        :type audios: Sequence[str | Path | bytes]
        :param num_speakers: Exact number of speakers, when it is known.
        :type num_speakers: int | None
        :param min_speakers: Lower bound on the speaker count.
        :type min_speakers: int | None
        :param max_speakers: Upper bound on the speaker count.
        :type max_speakers: int | None
        :param batch_size: Recordings processed concurrently per batch.
        :type batch_size: int
        :param desc: Progress bar description.
        :type desc: str | None
        :param continue_on_error: Keep going past a recording that failed,
            returning ``None`` in its place.
        :type continue_on_error: bool
        :returns: Speaker turns per recording, in input order, ``None`` for
            recordings that failed when `continue_on_error` is set.
        :rtype: List[List[SpeakerTurn] | None]
        :raises ValueError: If `batch_size` is not positive.
        """
        return await run_batched(
            audios,
            lambda audio: self.diarize(
                audio,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            ),
            batch_size=batch_size,
            desc=desc,
            continue_on_error=continue_on_error,
        )


@experimental
class DiarizationPyannote(Diarization):
    """
    Diarization through ``pyannote.audio``.

    The token is read from the environment (``HF_TOKEN`` by default).

    :param model: Model identifier on the HuggingFace hub.
    :type model: str
    :param device: Device string such as ``"cuda"`` or ``"cpu"``. ``None``
        leaves the pipeline on its default device.
    :type device: str | None
    :param token_variable: Environment variable holding the HuggingFace token.
        ``None`` searches :data:`HF_TOKEN_VARIABLES`.
    :type token_variable: str | None
    :param max_concurrency: Number of concurrent inference calls allowed.
    :type max_concurrency: int
    """

    def __init__(
        self,
        model: str = DEFAULT_PYANNOTE_MODEL,
        device: str | None = None,
        token_variable: str | None = None,
        max_concurrency: int = 1,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")

        self.model = model
        self.device = device
        self.token_variable = token_variable

        self._pipeline: Any = None
        self._load_lock = asyncio.Lock()
        self._inference_semaphore = asyncio.Semaphore(max_concurrency)

    async def diarize(
        self,
        audio: str | Path | bytes,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> List[SpeakerTurn]:
        """
        Find the speaker turns of a recording.

        :param audio: Path to an audio file, or its encoded bytes.
        :type audio: str | Path | bytes
        :param num_speakers: Exact number of speakers, when it is known.
        :type num_speakers: int | None
        :param min_speakers: Lower bound on the speaker count.
        :type min_speakers: int | None
        :param max_speakers: Upper bound on the speaker count.
        :type max_speakers: int | None
        :returns: Speaker turns in timeline order.
        :rtype: List[SpeakerTurn]
        :raises ImportError: If ``pyannote.audio`` is not installed.
        :raises RuntimeError: If no HuggingFace token is available.
        :raises ValueError: If the speaker bounds contradict each other.
        """
        if (
            min_speakers is not None
            and max_speakers is not None
            and min_speakers > max_speakers
        ):
            raise ValueError(
                f"min_speakers ({min_speakers}) is greater than max_speakers ({max_speakers})"
            )

        async with self._load_lock:
            if self._pipeline is None:
                await asyncio.to_thread(self._load)

        options: dict[str, Any] = {}
        if num_speakers is not None:
            options["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                options["min_speakers"] = min_speakers
            if max_speakers is not None:
                options["max_speakers"] = max_speakers

        if isinstance(audio, bytes):
            with tempfile.TemporaryDirectory(prefix="ragu-diarization-") as workdir:
                path = Path(workdir) / "audio.wav"
                path.write_bytes(audio)
                async with self._inference_semaphore:
                    return await asyncio.to_thread(self._diarize_sync, str(path), options)

        async with self._inference_semaphore:
            return await asyncio.to_thread(self._diarize_sync, str(audio), options)

    def _load(self) -> None:
        """
        Build the pyannote pipeline.

        :raises ImportError: If ``pyannote.audio`` is not installed.
        :raises RuntimeError: If no token is available or the model is not
            accessible with it.
        """
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is required for DiarizationPyannote. "
                "Install it with: pip install graph_ragu[local]"
            ) from exc

        token = self._read_token()
        pipeline = Pipeline.from_pretrained(self.model, **_token_kwargs(Pipeline, token))

        # from_pretrained returns None instead of raising when the token has
        # not accepted the model's terms, which is by far the most common
        # setup mistake with these checkpoints.
        if pipeline is None:
            raise RuntimeError(
                f"pyannote returned no pipeline for {self.model}. Accept the model's "
                f"terms at https://hf.co/{self.model} with the account owning the token."
            )

        if self.device:
            import torch

            pipeline.to(torch.device(self.device))

        self._pipeline = pipeline

    def _read_token(self) -> str:
        """
        Read the HuggingFace token from the environment.

        :returns: The token.
        :rtype: str
        :raises RuntimeError: If no token is set.
        """
        variables = (self.token_variable,) if self.token_variable else HF_TOKEN_VARIABLES

        for variable in variables:
            token = os.environ.get(variable)
            if token:
                return token

        raise RuntimeError(
            f"{self.model} is gated. Set {variables[0]} to a HuggingFace token whose "
            f"account has accepted the model's terms at https://hf.co/{self.model}."
        )

    def _diarize_sync(self, path: str, options: dict[str, Any]) -> List[SpeakerTurn]:
        """
        Run blocking inference.

        :param path: Path to the audio file.
        :type path: str
        :param options: Speaker-count options forwarded to the pipeline.
        :type options: dict[str, Any]
        :returns: Speaker turns in timeline order.
        :rtype: List[SpeakerTurn]
        :raises RuntimeError: If the pipeline returned something this adapter
            cannot read.
        """
        turns = parse_pyannote_output(self._pipeline(path, **options))
        turns.sort(key=lambda item: (item.start, item.end))
        return turns


def overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """
    Length of the intersection of two spans.

    :param start_a: Start of the first span.
    :type start_a: float
    :param end_a: End of the first span.
    :type end_a: float
    :param start_b: Start of the second span.
    :type start_b: float
    :param end_b: End of the second span.
    :type end_b: float
    :returns: Overlap in seconds; ``0.0`` when the spans are disjoint.
    :rtype: float
    """
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


class _TurnIndex:
    """
    Overlap lookup over a set of speaker turns.

    Turns are sorted once and queried with a bounded backward scan, so
    labelling tens of thousands of words against hundreds of turns stays linear
    in practice instead of quadratic. Overlapping turns are expected —
    diarizers emit them for simultaneous speech — and the longest overlap wins.

    :param turns: Speaker turns, in any order.
    :type turns: Sequence[SpeakerTurn]
    """

    def __init__(self, turns: Sequence[SpeakerTurn]) -> None:
        self.turns = sorted(turns, key=lambda turn: (turn.start, turn.end))
        self._starts = [turn.start for turn in self.turns]
        self._longest = max((turn.end - turn.start for turn in self.turns), default=0.0)

    def __bool__(self) -> bool:
        return bool(self.turns)

    def best(self, start: float, end: float) -> str | None:
        """
        Find the speaker overlapping a span the most.

        :param start: Start of the span in seconds.
        :type start: float
        :param end: End of the span in seconds.
        :type end: float
        :returns: Speaker label, or ``None`` when no turn overlaps the span.
        :rtype: str | None
        """
        if not self.turns:
            return None

        # Turns starting at or after `end` cannot overlap; from there, walk back
        # only as far as the longest turn could possibly reach.
        index = bisect_left(self._starts, end)
        horizon = start - self._longest

        best_speaker: str | None = None
        best_overlap = 0.0

        for position in range(index - 1, -1, -1):
            turn = self.turns[position]
            if turn.start < horizon:
                break

            shared = overlap(start, end, turn.start, turn.end)
            if shared > best_overlap:
                best_overlap = shared
                best_speaker = turn.speaker

        return best_speaker

    def nearest(self, start: float, end: float) -> str | None:
        """
        Find the speaker whose turn lies closest to a span.

        The fallback for words falling in a gap between turns, which happens at
        every turn boundary because the diarizer and the recognizer disagree
        slightly about where a word ends.

        :param start: Start of the span in seconds.
        :type start: float
        :param end: End of the span in seconds.
        :type end: float
        :returns: Speaker label, or ``None`` when there are no turns at all.
        :rtype: str | None
        """
        if not self.turns:
            return None

        index = bisect_left(self._starts, start)
        candidates = self.turns[max(0, index - 1): index + 2]

        def distance(turn: SpeakerTurn) -> float:
            if turn.end < start:
                return start - turn.end
            if turn.start > end:
                return turn.start - end
            return 0.0

        return min(candidates, key=distance).speaker


def _split_segment_by_speaker(segment: TranscriptSegment) -> List[TranscriptSegment]:
    """
    Cut a segment into runs of words sharing a speaker.

    :param segment: Segment whose words already carry speaker labels.
    :type segment: TranscriptSegment
    :returns: One segment per speaker run; the original segment unchanged when
        it has no words or only one speaker.
    :rtype: List[TranscriptSegment]
    """
    if not segment.words:
        return [segment]

    speakers = {word.speaker for word in segment.words}
    if len(speakers) <= 1:
        segment.speaker = next(iter(speakers))
        return [segment]

    pieces: list[TranscriptSegment] = []
    run: list[TranscriptWord] = []

    def flush() -> None:
        if not run:
            return
        pieces.append(
            TranscriptSegment(
                start=run[0].start,
                end=run[-1].end,
                text=join_word_texts(run),
                speaker=run[0].speaker,
                words=list(run),
            )
        )

    for word in segment.words:
        if run and word.speaker != run[-1].speaker:
            flush()
            run = []
        run.append(word)

    flush()
    return pieces


def merge_adjacent(
    segments: Sequence[TranscriptSegment],
    max_gap: float = 0.5,
    max_duration: float | None = 30.0,
) -> List[TranscriptSegment]:
    """
    Merge neighbouring segments that share a speaker.

    Word-level assignment produces one segment per speaker run, but a single
    speaker turn often arrives as several short runs split on pauses. Merging
    them back keeps the chunks a transcript chunker builds readable.

    The input is left alone: a merged segment is a new one, not a neighbour
    grown in place, so a caller's own segments do not change under them.

    :param segments: Segments in timeline order.
    :type segments: Sequence[TranscriptSegment]
    :param max_gap: Largest silence in seconds that may be bridged.
    :type max_gap: float
    :param max_duration: Do not grow a segment beyond this many seconds;
        ``None`` for no limit.
    :type max_duration: float | None
    :returns: Merged segments.
    :rtype: List[TranscriptSegment]
    """
    merged: list[TranscriptSegment] = []

    for segment in segments:
        if not merged:
            merged.append(segment.shifted(0.0))
            continue

        previous = merged[-1]
        fits = max_duration is None or segment.end - previous.start <= max_duration

        if not (
            previous.speaker == segment.speaker
            and segment.start - previous.end <= max_gap
            and fits
        ):
            merged.append(segment.shifted(0.0))
            continue

        words = [*previous.words, *segment.words]

        if previous.words and segment.words:
            text = join_word_texts(words)
        else:
            text = f"{previous.text} {segment.text}".strip()

        merged[-1] = TranscriptSegment(
            start=previous.start,
            end=max(previous.end, segment.end),
            text=text,
            speaker=previous.speaker,
            words=words,
        )

    return merged


def assign_speakers(
    transcript: Transcript,
    turns: Sequence[SpeakerTurn],
    strategy: Literal["word_overlap", "segment_overlap"] = "word_overlap",
    max_gap: float = 0.5,
    max_duration: float | None = 30.0,
) -> Transcript:
    """
    Marry diarization turns to a transcript so every segment has one speaker.

    ``word_overlap`` labels each word by the turn it overlaps most, then
    rebuilds segments so none of them spans a change of speaker. This is the
    accurate path and the reason word timings are worth requesting: a
    recognizer cuts on pauses and punctuation, so its segments straddle turn
    boundaries routinely.

    ``segment_overlap`` is the coarse fallback for backends that report no word
    timings. It labels whole segments and accepts that a segment covering two
    speakers keeps only the majority one.

    Segments that already carry a speaker are left alone, so a transcript from
    a diarizing ASR backend passes through unchanged.

    The result is always a new transcript with segments of its own, including
    when there is nothing to label. Returning the argument itself in that one
    case would make the result sometimes an alias and sometimes a copy, and a
    caller editing it would corrupt their input only on the recordings where
    the diarizer happened to find nobody.

    :param transcript: Transcript to label; not modified.
    :type transcript: Transcript
    :param turns: Speaker turns from a :class:`Diarization` backend.
    :type turns: Sequence[SpeakerTurn]
    :param strategy: ``word_overlap`` or ``segment_overlap``.
    :type strategy: Literal["word_overlap", "segment_overlap"]
    :param max_gap: Silence that may be bridged when re-merging neighbouring
        same-speaker segments.
    :type max_gap: float
    :param max_duration: Cap on merged segment length in seconds.
    :type max_duration: float | None
    :returns: A new transcript whose segments carry speakers.
    :rtype: Transcript
    :raises ValueError: If the strategy is unknown.
    """
    if strategy not in {"word_overlap", "segment_overlap"}:
        raise ValueError(
            f"Unknown strategy {strategy!r}; use 'word_overlap' or 'segment_overlap'"
        )

    index = _TurnIndex(turns)
    if not index or not transcript.segments:
        return Transcript(
            text=transcript.text,
            segments=[segment.shifted(0.0) for segment in transcript.segments],
            language=transcript.language,
            duration=transcript.duration,
            model=transcript.model,
        )

    if strategy == "word_overlap" and not any(segment.words for segment in transcript.segments):
        logger.warning(
            "Speaker assignment asked for word overlap, but the transcript carries no word "
            "timings; falling back to whole-segment overlap. Request word timestamps from "
            "the ASR backend for accurate speakers on segments that span a speaker change."
        )
        strategy = "segment_overlap"

    pieces: list[TranscriptSegment] = []

    for segment in transcript.segments:
        copy = segment.shifted(0.0)

        if copy.speaker is not None:
            pieces.append(copy)
            continue

        if strategy == "segment_overlap" or not copy.words:
            copy.speaker = (
                index.best(copy.start, copy.end) or index.nearest(copy.start, copy.end)
            )
            pieces.append(copy)
            continue

        for word in copy.words:
            word.speaker = (
                index.best(word.start, word.end) or index.nearest(word.start, word.end)
            )
        pieces.extend(_split_segment_by_speaker(copy))

    pieces.sort(key=lambda item: (item.start, item.end))
    merged = merge_adjacent(pieces, max_gap=max_gap, max_duration=max_duration)

    return Transcript(
        text=transcript.text,
        segments=[segment.shifted(0.0) for segment in merged],
        language=transcript.language,
        duration=transcript.duration,
        model=transcript.model,
    )