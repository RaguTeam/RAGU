import asyncio
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Sequence

from tqdm import tqdm

from ragu.common.batch_generator import BatchGenerator
from ragu.utils.batching import release_gpu_memory, run_batched
from ragu.common.logger import logger
from ragu.utils.audio import probe_duration, split_audio
from ragu.utils.ragu_utils import experimental

if TYPE_CHECKING:
    from ragu.models.openai import CachedAsyncOpenAI

DEFAULT_OPENAI_ASR_MODEL = "whisper-1"
DEFAULT_LOCAL_ASR_MODEL = "openai/whisper-large-v3"

MODELS_WITHOUT_TIMESTAMPS = frozenset({
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
})

DIARIZING_MODELS = frozenset({
    "gpt-4o-transcribe-diarize",
})

DEFAULT_MAX_UPLOAD_BYTES = 24 * 1024 * 1024

DEFAULT_BATCH_SIZE = 4


@dataclass(slots=True)
class TranscriptWord:
    """
    One recognized word with its own timings.

    Word timings are what makes speaker attribution accurate: a recognizer cuts
    segments on pauses and punctuation, so a segment routinely straddles a
    change of speaker. Only per-word bounds let
    :func:`~ragu.models.diarization.assign_speakers` split it in the right
    place.

    :param text: The word as produced by the backend, including its spacing.
    :param start: Start offset in seconds from the beginning of the recording.
    :param end: End offset in seconds.
    :param confidence: Backend confidence in ``[0, 1]``, when reported.
    :param speaker: Speaker label, filled in during speaker assignment.
    """

    text: str
    start: float
    end: float
    confidence: float | None = None
    speaker: str | None = None

    def shifted(self, offset: float) -> "TranscriptWord":
        """
        Return a copy moved along the timeline.

        :param offset: Seconds to add to both bounds.
        :type offset: float
        :returns: Shifted copy.
        :rtype: TranscriptWord
        """
        return TranscriptWord(
            text=self.text,
            start=self.start + offset,
            end=self.end + offset,
            confidence=self.confidence,
            speaker=self.speaker,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the word for storage in document metadata.

        :returns: JSON-serializable mapping.
        :rtype: dict[str, Any]
        """
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TranscriptWord":
        """
        Rebuild a word from its serialized form.

        :param payload: Mapping produced by :meth:`to_dict`.
        :type payload: dict[str, Any]
        :returns: The word.
        :rtype: TranscriptWord
        """
        return cls(
            text=payload.get("text") or "",
            start=float(payload["start"]),
            end=float(payload["end"]),
            confidence=payload.get("confidence"),
            speaker=payload.get("speaker"),
        )


@dataclass(slots=True)
class TranscriptSegment:
    """
    A timed span of recognized speech.

    :param start: Start offset in seconds from the beginning of the recording.
    :param end: End offset in seconds.
    :param text: Recognized text of the span.
    :param speaker: Speaker label, when the model performs diarization or a
        diarizer was merged in afterwards.
    :param words: Word-level timings, when the backend was asked for them and
        supports them. Empty otherwise, which downgrades speaker assignment to
        whole-segment overlap.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None
    words: List["TranscriptWord"] = field(default_factory=list)

    def shifted(self, offset: float) -> "TranscriptSegment":
        """
        Return a copy moved along the timeline.

        :param offset: Seconds to add to both bounds.
        :type offset: float
        :returns: Shifted copy.
        :rtype: TranscriptSegment
        """
        return TranscriptSegment(
            start=self.start + offset,
            end=self.end + offset,
            text=self.text,
            speaker=self.speaker,
            words=[word.shifted(offset) for word in self.words],
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the segment for storage in document metadata.

        ``words`` is omitted when empty, so documents produced without word
        timings keep the metadata shape they had before.

        :returns: JSON-serializable mapping.
        :rtype: dict[str, Any]
        """
        payload: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
        }
        if self.words:
            payload["words"] = [word.to_dict() for word in self.words]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TranscriptSegment":
        """
        Rebuild a segment from its serialized form.

        ``words`` is optional, so a document written before word timings were
        requested — or by a backend that reports none — reads back cleanly.

        :param payload: Mapping produced by :meth:`to_dict`.
        :type payload: dict[str, Any]
        :returns: The segment, carrying its words when they were stored.
        :rtype: TranscriptSegment
        """
        return cls(
            start=float(payload["start"]),
            end=float(payload["end"]),
            text=payload.get("text") or "",
            speaker=payload.get("speaker"),
            words=[TranscriptWord.from_dict(word) for word in payload.get("words") or ()],
        )


@dataclass(slots=True)
class Transcript:
    """
    Result of transcribing one recording.

    ``segments`` may be empty: several hosted models return the transcript as
    a single blob with no timestamps, so consumers must treat timed output as
    optional and fall back to ``text``.

    :param text: Full transcript text.
    :param segments: Timed segments, when the model provides them.
    :param language: Detected or requested language code.
    :param duration: Length of the recording in seconds, when known.
    :param model: Identifier of the model that produced the transcript.
    """

    text: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the whole transcript.

        :returns: JSON-serializable mapping.
        :rtype: dict[str, Any]
        """
        return {
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
            "language": self.language,
            "duration": self.duration,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Transcript":
        """
        Rebuild a transcript from its serialized form.

        The counterpart of :meth:`to_dict`, and of the per-segment mappings a
        parser stores under ``metadata["segments"]``: a consumer reading those
        back gets objects rather than having to walk raw dictionaries itself.

        :param payload: Mapping produced by :meth:`to_dict`.
        :type payload: dict[str, Any]
        :returns: The transcript.
        :rtype: Transcript
        """
        return cls(
            text=payload.get("text") or "",
            segments=[
                TranscriptSegment.from_dict(segment)
                for segment in payload.get("segments") or ()
            ],
            language=payload.get("language"),
            duration=payload.get("duration"),
            model=payload.get("model") or "",
        )


class ASR(ABC):
    """
    Abstract interface for speech-to-text backends.

    Implementations turn an audio file into a :class:`Transcript`. Video is
    not handled here: callers extract the audio track first (see
    :func:`ragu.utils.audio.extract_audio`).
    """

    @abstractmethod
    async def transcribe(
        self,
        audio: str | Path | bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        """
        Transcribe a single recording.

        :param audio: Path to an audio file, or its encoded bytes.
        :type audio: str | Path | bytes
        :param language: Language code hint (e.g. ``"ru"``); ``None`` lets the
            backend detect it.
        :type language: str | None
        :param prompt: Optional prompt biasing the decoder (terminology, names).
        :type prompt: str | None
        :returns: Recognized transcript.
        :rtype: Transcript
        """

    async def transcribe_batch(
        self,
        audios: Sequence[str | Path | bytes],
        language: str | None = None,
        prompt: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        desc: str | None = "Transcribing",
        continue_on_error: bool = False,
    ) -> List[Transcript | None]:
        """
        Transcribe several recordings, `batch_size` at a time.

        Recordings within a batch run concurrently; batches run one after the
        other. Handing the whole list to :func:`asyncio.gather` instead would
        open one request — or one decode — per recording at once, which is how
        a hosted backend hits its rate limit and a local one runs out of memory.

        :param audios: Audio paths or encoded bytes.
        :type audios: Sequence[str | Path | bytes]
        :param language: Language code hint applied to every recording.
        :type language: str | None
        :param prompt: Optional prompt applied to every recording.
        :type prompt: str | None
        :param batch_size: Recordings processed concurrently per batch.
        :type batch_size: int
        :param desc: Progress bar description; transcription is slow enough
            that a corpus should not look hung while it runs.
        :type desc: str | None
        :param continue_on_error: Keep going past a recording that failed,
            returning ``None`` in its place, instead of losing the whole run to
            one unreadable file.
        :type continue_on_error: bool
        :returns: Transcripts in input order, ``None`` for recordings that
            failed when `continue_on_error` is set.
        :rtype: List[Transcript | None]
        :raises ValueError: If `batch_size` is not positive.
        """
        return await run_batched(
            audios,
            lambda audio: self.transcribe(audio, language=language, prompt=prompt),
            batch_size=batch_size,
            desc=desc,
            continue_on_error=continue_on_error,
        )


SENTENCE_ENDINGS = ".!?…。！？"
CLINGING_PUNCTUATION = ".,!?;:)]}»…%‰°\"'"
OPENING_PUNCTUATION = "([{«"


def join_word_texts(words: Sequence[TranscriptWord]) -> str:
    """
    Reassemble the text of a run of words.

    Backends disagree about spacing. faster-whisper and the OpenAI API emit
    tokens carrying their own leading space (``" Редактор"``, ``".Кулак"``), so
    concatenating them verbatim reproduces the original text exactly. The
    ``transformers`` pipeline strips that spacing, leaving bare tokens that
    have to be rejoined — and naively joining those with spaces produces
    ``"А .Кулак"``, so punctuation is reattached to its neighbour.

    :param words: Words to join, in order.
    :type words: Sequence[TranscriptWord]
    :returns: The reassembled text.
    :rtype: str
    """
    if not words:
        return ""

    if any(word.text[:1].isspace() for word in words):
        return "".join(word.text for word in words).strip()

    parts: list[str] = []
    for word in words:
        token = word.text.strip()
        if not token:
            continue

        clings_back = token[0] in CLINGING_PUNCTUATION
        opens = bool(parts) and parts[-1][-1:] in OPENING_PUNCTUATION

        if parts and not clings_back and not opens:
            parts.append(" ")
        parts.append(token)

    return "".join(parts).strip()


def group_words_into_segments(
    words: Sequence[TranscriptWord],
    max_gap: float = 0.6,
    max_duration: float = 30.0,
) -> List[TranscriptSegment]:
    """
    Build segments out of a flat stream of words.

    Backends asked for word timings report no segmentation of their own, but
    everything downstream — chunking, speaker assignment, citation — works in
    segments. Cutting on a pause, on a sentence ending, or on a length cap
    reproduces roughly what a recognizer's own segmenter does.

    :param words: Words in timeline order.
    :type words: Sequence[TranscriptWord]
    :param max_gap: Silence between words that starts a new segment.
    :type max_gap: float
    :param max_duration: Longest segment produced, in seconds.
    :type max_duration: float
    :returns: Segments carrying their words.
    :rtype: List[TranscriptSegment]
    """
    segments: list[TranscriptSegment] = []
    current: list[TranscriptWord] = []

    def flush() -> None:
        if not current:
            return
        segments.append(
            TranscriptSegment(
                start=current[0].start,
                end=current[-1].end,
                text=join_word_texts(current),
                words=list(current),
            )
        )

    for word in words:
        if current:
            gap = word.start - current[-1].end
            grown = word.end - current[0].start
            ends_sentence = current[-1].text.strip().endswith(tuple(SENTENCE_ENDINGS))

            if gap > max_gap or grown > max_duration or (ends_sentence and grown > 1.0):
                flush()
                current = []

        current.append(word)

    flush()
    return segments


def _is_response_format_error(error: Any, attempted: str) -> bool:
    """
    Decide whether a rejected request blames the response format.

    Told apart from an unsupported language or a malformed upload, which
    downgrading would retry identically while reporting the wrong cause.

    OpenAI names the offending field in ``param``. Other servers do not: vLLM
    answers ``Currently do not support diarized_json for <model>`` with
    ``param`` unset, naming the *format* rather than the field it arrived in.
    So the format this call actually asked for counts as evidence too — it is
    specific enough not to match a complaint about anything else.

    :param error: The ``BadRequestError`` the provider raised.
    :type error: Any
    :param attempted: Response format the rejected request asked for.
    :type attempted: str
    :returns: ``True`` when the format is what was rejected.
    :rtype: bool
    """
    parameter = getattr(error, "param", None)
    if parameter:
        return parameter in ("response_format", "timestamp_granularities")

    message = str(error)
    return (
        attempted in message
        or "response_format" in message
        or "timestamp_granularities" in message
    )


def _audio_name(audio: str | Path | bytes) -> str:
    """
    Name the recording for the provider, which reads the container off it.

    :param audio: Path to an audio file, or its encoded bytes.
    :type audio: str | Path | bytes
    :returns: File name to report to the provider.
    :rtype: str
    """
    return "audio.wav" if isinstance(audio, bytes) else Path(audio).name


def _audio_size(audio: str | Path | bytes) -> int:
    """
    Measure the recording without reading it.

    Deciding whether to split is a question about the file's size, and answering
    it by loading a three-hour recording into memory would defeat the splitting
    it decides on.

    :param audio: Path to an audio file, or its encoded bytes.
    :type audio: str | Path | bytes
    :returns: Size in bytes.
    :rtype: int
    """
    return len(audio) if isinstance(audio, bytes) else Path(audio).stat().st_size


def _pipeline_payload(audio: str | Path | bytes) -> str | bytes:
    """
    Normalize a recording into what the transformers pipeline accepts.

    :param audio: Path to an audio file, or its encoded bytes.
    :type audio: str | Path | bytes
    :returns: The path as a string, or the bytes unchanged.
    :rtype: str | bytes
    """
    return str(audio) if isinstance(audio, (str, Path)) else audio


def _read_audio(audio: str | Path | bytes) -> bytes:
    """
    Read the recording into memory for upload.

    :param audio: Path to an audio file, or its encoded bytes.
    :type audio: str | Path | bytes
    :returns: Audio bytes.
    :rtype: bytes
    """
    return audio if isinstance(audio, bytes) else Path(audio).read_bytes()


@experimental
class ASROpenAI(ASR):
    """
    Speech-to-text over an OpenAI-compatible API.

    Two provider quirks are handled here:

    - Response format depends on the model. ``whisper-1`` supports
      ``verbose_json`` with segment timestamps, ``gpt-4o-transcribe`` and its
      mini variant support only ``json`` (no timestamps at all), and
      ``gpt-4o-transcribe-diarize`` returns speaker-labelled segments through
      ``diarized_json``. Picking a model without timestamps is legitimate but
      silently disables time-aware chunking, so it is logged as a warning.
      The table holds OpenAI's own names, so a model served under another one
      — an Azure deployment, a vLLM server — may be asked for a format it does
      not support; the first rejection downgrades the instance to plain
      ``json`` instead of failing.
    - Uploads are capped at 25 MB. Longer recordings are split with ffmpeg and
      stitched back into one timeline.

    Example::

        from ragu.models.openai import CachedAsyncOpenAI
        from ragu.models.asr import ASROpenAI

        client = CachedAsyncOpenAI(api_key="...")
        asr = ASROpenAI(client=client, model_name="whisper-1")

    :param client: OpenAI-compatible client used for transcription calls.
    :type client: CachedAsyncOpenAI
    :param model_name: Transcription model registered on the endpoint.
    :type model_name: str
    :param response_format: ``"auto"`` picks the richest format the model
        supports; an explicit value is passed through unchanged.
    :type response_format: str
    :param timestamp_granularities: Granularities requested alongside
        ``verbose_json``. Words are requested by default: they cost nothing
        extra and are what makes speaker assignment accurate. Pass
        ``("segment",)`` to skip them.
    :type timestamp_granularities: Sequence[str]
    :param language: Default language hint.
    :type language: str | None
    :param prompt: Default decoder prompt.
    :type prompt: str | None
    :param max_upload_bytes: Split recordings larger than this.
    :type max_upload_bytes: int
    :param split_seconds: Length of each part when splitting.
    :type split_seconds: float
    :param max_parallel_parts: Parts of one split recording uploaded at once.
    :type max_parallel_parts: int
    :param create_kwargs: Extra options forwarded to the transcription call.
    """

    def __init__(
        self,
        client: "CachedAsyncOpenAI",
        model_name: str = DEFAULT_OPENAI_ASR_MODEL,
        response_format: str = "auto",
        timestamp_granularities: Sequence[str] = ("segment", "word"),
        language: str | None = None,
        prompt: str | None = None,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        split_seconds: float = 600.0,
        max_parallel_parts: int = DEFAULT_BATCH_SIZE,
        **create_kwargs: Any,
    ) -> None:
        if max_parallel_parts < 1:
            raise ValueError(
                f"max_parallel_parts must be positive, got {max_parallel_parts}"
            )

        self.client = client
        self.model_name = model_name
        self.response_format = response_format
        self.timestamp_granularities = tuple(timestamp_granularities)
        self.language = language
        self.prompt = prompt
        self.max_upload_bytes = max_upload_bytes
        self.split_seconds = split_seconds
        self.max_parallel_parts = max_parallel_parts
        self.create_kwargs = create_kwargs

        self._warned_about_timestamps = False
        self._warned_about_words = False
        self._downgraded_format: str | None = None

    async def transcribe(
        self,
        audio: str | Path | bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        """
        Transcribe a recording, splitting it when it exceeds the upload limit.

        :param audio: Path to an audio file, or its encoded bytes.
        :type audio: str | Path | bytes
        :param language: Language hint; falls back to the client default.
        :type language: str | None
        :param prompt: Decoder prompt; falls back to the client default.
        :type prompt: str | None
        :returns: Recognized transcript.
        :rtype: Transcript
        :raises RuntimeError: If splitting is needed but ffmpeg is missing.
        """
        name = _audio_name(audio)
        size = _audio_size(audio)

        if size <= self.max_upload_bytes:
            payload = await self._request(_read_audio(audio), name, language, prompt)
            return self._to_transcript(payload)

        logger.info(
            "Audio {} is {:.1f} MB, above the {:.1f} MB upload limit; "
            "splitting into {:.0f}s parts.",
            name,
            size / 1024 / 1024,
            self.max_upload_bytes / 1024 / 1024,
            self.split_seconds,
        )
        return await self._transcribe_split(audio, name, language, prompt)

    async def _transcribe_split(
        self,
        audio: str | Path | bytes,
        name: str,
        language: str | None,
        prompt: str | None,
    ) -> Transcript:
        """
        Split an oversized recording, transcribe the parts and stitch them.

        The source is never held in memory as a whole: ffmpeg reads it from disk
        and only the parts, which are under the upload limit by construction,
        are read back.

        :param audio: Original input, reused when it is already a file.
        :param name: File name used for the temporary copy of in-memory input.
        :param language: Language hint.
        :param prompt: Decoder prompt.
        :returns: Transcript covering the whole recording.
        :rtype: Transcript
        """
        with tempfile.TemporaryDirectory(prefix="ragu-asr-") as workdir:
            if isinstance(audio, bytes):
                source = Path(workdir) / name
                source.write_bytes(audio)
            else:
                source = Path(audio)

            parts = await split_audio(source, self.split_seconds, Path(workdir) / "parts")
            self._check_parts(parts, name)

            payloads = await run_batched(
                parts,
                lambda part: self._request(part[0].read_bytes(), part[0].name, language, prompt),
                batch_size=self.max_parallel_parts,
                desc=f"Transcribing {name} in parts",
            )

        transcripts = [
            self._to_transcript(payload, offset=offset)
            for payload, (_part, offset) in zip(payloads, parts)
        ]
        return self._merge(transcripts)

    def _check_parts(self, parts: Sequence[tuple[Path, float]], name: str) -> None:
        """
        Refuse a split the provider could not be asked to transcribe.

        Two ways it can go wrong. A split that produced nothing at all would
        stitch into an empty transcript, which reads exactly like a silent
        recording — the one failure worth being loud about. And splitting is
        decided in bytes but performed in seconds, so a `split_seconds` too
        generous for the recording's bitrate leaves parts still over the limit;
        failing here names the setting to change, where the provider's own 400
        would not.

        :param parts: ``(path, offset)`` pairs produced by the split.
        :type parts: Sequence[tuple[Path, float]]
        :param name: Name of the recording, for the message.
        :type name: str
        :raises RuntimeError: If the split produced no parts.
        :raises ValueError: If any part is still above `max_upload_bytes`.
        """
        if not parts:
            raise RuntimeError(
                f"Splitting {name} produced no parts. The recording was not transcribed; "
                f"check that ffmpeg could read it."
            )

        largest = max(part.stat().st_size for part, _offset in parts)
        if largest <= self.max_upload_bytes:
            return

        raise ValueError(
            f"Splitting {name} into {self.split_seconds:.0f}s parts still leaves one of "
            f"{largest / 1024 / 1024:.1f} MB, above the "
            f"{self.max_upload_bytes / 1024 / 1024:.1f} MB upload limit. "
            f"Lower split_seconds."
        )

    async def _request(
        self,
        data: bytes,
        name: str,
        language: str | None,
        prompt: str | None,
    ) -> dict[str, Any]:
        """
        Send one transcription request.

        Retried once with a plain ``json`` format when the endpoint rejects the
        rich one this adapter guessed at — see :meth:`_downgrade_format`.

        :param data: Audio bytes to upload.
        :param name: File name reported to the provider.
        :param language: Language hint.
        :param prompt: Decoder prompt.
        :returns: Raw provider payload.
        :rtype: dict[str, Any]
        :raises openai.BadRequestError: If the request is rejected for a reason
            other than the response format, or after the format was downgraded.
        """
        from openai import BadRequestError

        try:
            return await self._send(data, name, language, prompt)
        except BadRequestError as error:
            if not self._downgrade_format(error):
                raise

        return await self._send(data, name, language, prompt)

    async def _send(
        self,
        data: bytes,
        name: str,
        language: str | None,
        prompt: str | None,
    ) -> dict[str, Any]:
        """
        Build and send one transcription request.

        :param data: Audio bytes to upload.
        :param name: File name reported to the provider.
        :param language: Language hint.
        :param prompt: Decoder prompt.
        :returns: Raw provider payload.
        :rtype: dict[str, Any]
        """
        response_format = self._resolve_response_format()
        options: dict[str, Any] = {"response_format": response_format, **self.create_kwargs}

        if response_format == "verbose_json" and self.timestamp_granularities:
            options["timestamp_granularities"] = list(self.timestamp_granularities)

        effective_language = language if language is not None else self.language
        if effective_language:
            options["language"] = effective_language

        effective_prompt = prompt if prompt is not None else self.prompt
        if effective_prompt:
            options["prompt"] = effective_prompt

        return await self.client.transcribe(
            model_name=self.model_name,
            audio=data,
            audio_name=name,
            **options,
        )

    def _downgrade_format(self, error: Any) -> bool:
        """
        Fall back to plain ``json`` after the endpoint rejected a rich format.

        The format is chosen from a table of OpenAI model names, which is right
        for OpenAI and a guess everywhere else: an Azure deployment, a vLLM
        server or a proxy names its models whatever it likes, and a model that
        cannot do ``verbose_json`` under an unfamiliar name gets asked for it
        anyway and answers with a 400. Rather than making the caller discover
        that, the first rejection downgrades the instance for good.

        Only an automatic choice is downgraded. An explicit `response_format`
        is the caller's instruction, and quietly overriding it would hide their
        own mistake from them.

        :param error: The rejection, inspected to tell a format the endpoint
            does not support from a rejection about something else entirely.
        :type error: Any
        :returns: ``True`` if the format was downgraded and the call is worth
            retrying, ``False`` if there is nothing left to try.
        :rtype: bool
        """
        if self.response_format != "auto" or self._downgraded_format is not None:
            return False

        current = self._resolve_response_format()
        if current == "json":
            # Nothing richer was asked for, so the rejection is about something
            # else and the retry would send an identical request.
            return False

        if not _is_response_format_error(error, current):
            return False

        logger.warning(
            "{} rejected the '{}' response format; falling back to plain 'json'. "
            "Transcripts from this model will have no segments, so time-aware "
            "chunking falls back to plain text. Pass response_format explicitly "
            "to silence this.",
            self.model_name,
            current,
        )
        self._downgraded_format = "json"
        return True

    def _resolve_response_format(self) -> str:
        """
        Pick the response format for the configured model.

        :returns: Response format accepted by the model.
        :rtype: str
        """
        if self.response_format != "auto":
            return self.response_format

        if self._downgraded_format is not None:
            return self._downgraded_format

        if self.model_name in DIARIZING_MODELS:
            return "diarized_json"

        if self.model_name in MODELS_WITHOUT_TIMESTAMPS:
            if not self._warned_about_timestamps:
                logger.warning(
                    "Model {} does not support segment timestamps; transcripts "
                    "will have no segments and time-aware chunking will fall "
                    "back to plain text. Use 'whisper-1' for timestamps or "
                    "'gpt-4o-transcribe-diarize' for speakers.",
                    self.model_name,
                )
                self._warned_about_timestamps = True
            return "json"

        return "verbose_json"

    def _to_transcript(self, payload: dict[str, Any], offset: float = 0.0) -> Transcript:
        """
        Map a provider payload onto a :class:`Transcript`.

        :param payload: Raw provider payload.
        :param offset: Seconds to add to every timestamp (used when stitching).
        :returns: Parsed transcript.
        :rtype: Transcript
        """
        raw_segments = [
            raw for raw in payload.get("segments") or ()
            if raw.get("start") is not None and raw.get("end") is not None
        ]
        words_by_segment = self._distribute_words(payload, raw_segments, offset)

        segments: list[TranscriptSegment] = []
        for index, raw in enumerate(raw_segments):
            segments.append(
                TranscriptSegment(
                    start=float(raw["start"]) + offset,
                    end=float(raw["end"]) + offset,
                    text=(raw.get("text") or "").strip(),
                    speaker=raw.get("speaker"),
                    words=words_by_segment.get(index, []),
                )
            )

        self._check_word_timings(segments)

        duration = payload.get("duration")
        return Transcript(
            text=(payload.get("text") or "").strip(),
            segments=segments,
            language=payload.get("language"),
            duration=float(duration) if duration is not None else None,
            model=self.model_name,
        )

    def _check_word_timings(self, segments: Sequence[TranscriptSegment]) -> None:
        """
        Report once that the word timings asked for did not arrive.

        An endpoint is free to accept ``timestamp_granularities`` and answer
        without any words — vLLM does exactly that, returning ``"words": null``
        beside perfectly good segments. Nothing fails, which is the problem:
        speaker assignment quietly drops to whole-segment overlap, and a segment
        spanning a change of speaker is attributed wholly to one of them.

        :param segments: Segments parsed from the payload.
        :type segments: Sequence[TranscriptSegment]
        """
        if self._warned_about_words or "word" not in self.timestamp_granularities:
            return

        # No segments at all is a different problem, already reported elsewhere.
        if not segments or any(segment.words for segment in segments):
            return

        logger.warning(
            "{} accepted the request for word timestamps but returned none. Speaker "
            "assignment will fall back to whole-segment overlap, which mis-attributes "
            "any segment spanning a change of speaker. Pass "
            "timestamp_granularities=('segment',) to silence this.",
            self.model_name,
        )
        self._warned_about_words = True

    @staticmethod
    def _distribute_words(
        payload: dict[str, Any],
        raw_segments: Sequence[dict[str, Any]],
        offset: float,
    ) -> dict[int, list[TranscriptWord]]:
        """
        Assign the payload's flat word list to the segments it belongs to.

        ``verbose_json`` reports words at the top level rather than nested in
        segments, so each word is placed by which segment its midpoint falls
        into. The midpoint rather than the start: a word straddling a segment
        boundary belongs to whichever segment holds most of it.

        :param payload: Raw provider payload.
        :type payload: dict[str, Any]
        :param raw_segments: The payload's segments, already filtered to those
            carrying bounds.
        :type raw_segments: Sequence[dict[str, Any]]
        :param offset: Seconds to add to every timestamp.
        :type offset: float
        :returns: Words keyed by their segment's index.
        :rtype: dict[int, list[TranscriptWord]]
        """
        raw_words = payload.get("words") or ()
        if not raw_words or not raw_segments:
            return {}

        bounds = [(float(raw["start"]), float(raw["end"])) for raw in raw_segments]
        buckets: dict[int, list[TranscriptWord]] = {}
        cursor = 0

        for raw in raw_words:
            start, end = raw.get("start"), raw.get("end")
            if start is None or end is None:
                continue

            midpoint = (float(start) + float(end)) / 2
            while cursor + 1 < len(bounds) and midpoint > bounds[cursor][1]:
                cursor += 1

            buckets.setdefault(cursor, []).append(
                TranscriptWord(
                    text=raw.get("word") or raw.get("text") or "",
                    start=float(start) + offset,
                    end=float(end) + offset,
                )
            )

        return buckets

    def _merge(self, transcripts: Iterable[Transcript]) -> Transcript:
        """
        Combine per-part transcripts into a single timeline.

        :param transcripts: Transcripts of consecutive parts, already shifted.
        :returns: Merged transcript.
        :rtype: Transcript
        """
        parts = list(transcripts)
        segments = [segment for part in parts for segment in part.segments]
        texts = [part.text for part in parts if part.text]

        duration: float | None = None
        if segments:
            duration = max(segment.end for segment in segments)

        language = next((part.language for part in parts if part.language), None)

        return Transcript(
            text=" ".join(texts),
            segments=segments,
            language=language,
            duration=duration,
            model=self.model_name,
        )


@experimental
class ASRTransformers(ASR):
    """
    Local speech-to-text through the HuggingFace ``transformers`` pipeline.

    Requires the ``local`` extra (``pip install graph_ragu[local]``) and the
    ``ffmpeg`` binary for anything that is not plain WAV. 

    :param model_name_or_path: HF model id or local checkpoint path.
    :type model_name_or_path: str
    :param device: Device string passed to the pipeline (e.g. ``"cuda:0"``,
        ``"cpu"``). ``None`` lets transformers decide.
    :type device: str | int | None
    :param dtype: ``dtype`` passed to the pipeline.
    :type dtype: str
    :param chunk_length_s: Window length used for long-form decoding.
    :type chunk_length_s: float
    :param batch_size: Batch size for chunked decoding.
    :type batch_size: int
    :param return_timestamps: ``True`` for segment timestamps, ``"word"`` for
        word-level ones, ``False`` to skip them.
    :type return_timestamps: bool | str
    :param language: Default language hint passed to ``generate_kwargs``.
    :type language: str | None
    :param prompt: Default decoder prompt, applied where the backend can take
        one. See :meth:`_prompt_ids`.
    :type prompt: str | None
    :param task: Whisper task, ``"transcribe"`` or ``"translate"``.
    :type task: str
    :param max_concurrency: Number of concurrent inference calls allowed.
    :type max_concurrency: int
    :param generate_kwargs: Extra generation options.
    :type generate_kwargs: dict[str, Any] | None
    :param pipeline_kwargs: Extra options forwarded to ``pipeline(...)``.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_LOCAL_ASR_MODEL,
        device: str | int | None = None,
        dtype: str = "auto",
        chunk_length_s: float = 30.0,
        batch_size: int = 1,
        return_timestamps: bool | str = True,
        language: str | None = None,
        prompt: str | None = None,
        task: str = "transcribe",
        max_concurrency: int = 1,
        generate_kwargs: dict[str, Any] | None = None,
        **pipeline_kwargs: Any,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        self.return_timestamps = return_timestamps
        self.language = language
        self.prompt = prompt
        self.task = task
        self.generate_kwargs = dict(generate_kwargs) if generate_kwargs else {}
        self.pipeline_kwargs = pipeline_kwargs

        self._pipeline: Any = None
        self._load_lock = asyncio.Lock()
        self._inference_semaphore = asyncio.Semaphore(max_concurrency)
        self._warned_about_prompt = False
        self._prompt_ids_cache: dict[tuple[str, str], Any] = {}

    async def transcribe(
        self,
        audio: str | Path | bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        """
        Transcribe a recording with the local model.

        :param audio: Path to an audio file, or its encoded bytes.
        :type audio: str | Path | bytes
        :param language: Language hint; falls back to the instance default.
        :type language: str | None
        :param prompt: Terminology or names biasing the decoder. Applied on
            backends whose tokenizer can build Whisper prompt tokens; reported
            once and ignored on those that cannot.
        :type prompt: str | None
        :returns: Recognized transcript.
        :rtype: Transcript
        :raises ImportError: If ``transformers`` is not installed.
        """
        await self._ensure_loaded()

        duration = await self._duration(audio)

        async with self._inference_semaphore:
            output = await asyncio.to_thread(
                self._transcribe_sync, audio, language, prompt
            )

        return self._to_transcript(output, duration)

    async def transcribe_batch(
        self,
        audios: Sequence[str | Path | bytes],
        language: str | None = None,
        prompt: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        desc: str | None = "Transcribing",
        continue_on_error: bool = False,
    ) -> List[Transcript | None]:
        """
        Transcribe several recordings, one pipeline call per batch.

        :param audios: Audio paths or encoded bytes.
        :type audios: Sequence[str | Path | bytes]
        :param language: Language code hint applied to every recording.
        :type language: str | None
        :param prompt: Decoder prompt applied to every recording, as in
            :meth:`transcribe`.
        :type prompt: str | None
        :param batch_size: Recordings per pipeline call. Raise it for
            throughput, lower it when VRAM is tight.
        :type batch_size: int
        :param desc: Progress bar description.
        :type desc: str | None
        :param continue_on_error: Keep going past a recording that failed,
            returning ``None`` in its place. A batch that fails as a whole is
            retried one recording at a time, so neither a single unreadable file
            nor a batch too wide for the card costs everything it was grouped
            with — the latter recovers in full, since the recordings themselves
            were fine.
        :type continue_on_error: bool
        :returns: Transcripts in input order, ``None`` for recordings that
            failed when `continue_on_error` is set.
        :rtype: List[Transcript | None]
        :raises ValueError: If `batch_size` is not positive.
        :raises ImportError: If ``transformers`` is not installed.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        await self._ensure_loaded()

        recordings = list(audios)
        durations = await asyncio.gather(*(self._duration(audio) for audio in recordings))
        timed = list(zip(recordings, durations))

        results: List[Transcript | None] = []
        with tqdm(total=len(timed), desc=desc) as progress:
            for batch in BatchGenerator(timed, batch_size).get_batches():
                results.extend(await self._transcribe_group(
                    batch, language, prompt, continue_on_error, progress,
                ))

        return results

    async def _transcribe_group(
        self,
        batch: Sequence[tuple[str | Path | bytes, float | None]],
        language: str | None,
        prompt: str | None,
        continue_on_error: bool,
        progress: Any,
    ) -> List[Transcript | None]:
        """
        Put one batch through a single pipeline call.

        :param batch: ``(audio, duration)`` pairs forming the batch.
        :type batch: Sequence[tuple[str | Path | bytes, float | None]]
        :param language: Language hint.
        :type language: str | None
        :param prompt: Decoder prompt.
        :type prompt: str | None
        :param continue_on_error: Fall back to one recording at a time when the
            batch fails as a whole, instead of raising.
        :type continue_on_error: bool
        :param progress: Progress bar to advance.
        :type progress: Any
        :returns: Transcripts in batch order.
        :rtype: List[Transcript | None]
        """
        payloads = [_pipeline_payload(audio) for audio, _duration in batch]
        outputs: List[Any] | None = None
        failure: str | None = None

        try:
            async with self._inference_semaphore:
                outputs = await asyncio.to_thread(
                    self._transcribe_many_sync, payloads, language, prompt
                )
        except Exception as error:  # noqa: BLE001 - the policy decides what this means
            if not continue_on_error:
                raise
            failure = f"{type(error).__name__}: {error}"

        if failure is not None:
            logger.warning(
                "A batch of {} recordings failed ({}); retrying them one at a time",
                len(batch), failure,
            )
            await asyncio.to_thread(release_gpu_memory)
            return await self._transcribe_singly(batch, language, prompt, progress)

        progress.update(len(batch))
        return [
            self._to_transcript(output, duration)
            for output, (_audio, duration) in zip(outputs or (), batch)
        ]

    async def _transcribe_singly(
        self,
        batch: Sequence[tuple[str | Path | bytes, float | None]],
        language: str | None,
        prompt: str | None,
        progress: Any,
    ) -> List[Transcript | None]:
        """
        Transcribe a failed batch one recording at a time.

        :param batch: ``(audio, duration)`` pairs to retry.
        :type batch: Sequence[tuple[str | Path | bytes, float | None]]
        :param language: Language hint.
        :type language: str | None
        :param prompt: Decoder prompt.
        :type prompt: str | None
        :param progress: Progress bar to advance.
        :type progress: Any
        :returns: Transcripts in batch order, ``None`` where one failed.
        :rtype: List[Transcript | None]
        """
        results: List[Transcript | None] = []

        for audio, duration in batch:
            try:
                async with self._inference_semaphore:
                    output = await asyncio.to_thread(
                        self._transcribe_sync, audio, language, prompt
                    )
                results.append(self._to_transcript(output, duration))
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                logger.warning(
                    "Transcription failed for {}: {}: {}",
                    _audio_name(audio), type(error).__name__, error,
                )
                results.append(None)
            finally:
                progress.update(1)

        return results

    async def _ensure_loaded(self) -> None:
        """
        Load the pipeline on first use.

        :raises ImportError: If ``transformers`` is not installed.
        """
        async with self._load_lock:
            if self._pipeline is None:
                await asyncio.to_thread(self._load)

    def _prompt_ids(self, prompt: str | None) -> Any:
        """
        Turn a prompt into the decoder tokens Whisper conditions on.

        Tokens are cached: a corpus transcribed under one prompt would
        otherwise tokenize the same string per recording.

        :param prompt: Terminology or names to bias the decoder with.
        :return: Prompt tokens on the model's device, or ``None`` when the
            backend cannot take them.
        """
        if not prompt:
            return None

        tokenizer = getattr(self._pipeline, "tokenizer", None)
        build = getattr(tokenizer, "get_prompt_ids", None)
        if build is None:
            self._warn_about_prompt()
            return None

        device = str(getattr(getattr(self._pipeline, "model", None), "device", "cpu"))
        key = (prompt, device)
        if key not in self._prompt_ids_cache:
            self._prompt_ids_cache[key] = build(prompt, return_tensors="pt").to(device)

        return self._prompt_ids_cache[key]

    def _warn_about_prompt(self) -> None:
        """
        Report once that this backend cannot take a decoder prompt.
        """
        if self._warned_about_prompt:
            return

        logger.warning(
            "{} cannot apply a decoder prompt: {} has no tokenizer able to build "
            "Whisper prompt tokens. A CTC model has no decoder to condition; for "
            "the rest, bias generation through generate_kwargs instead.",
            type(self).__name__,
            self.model_name_or_path,
        )
        self._warned_about_prompt = True

    def _load(self) -> None:
        """
        Build the transformers ASR pipeline.

        :raises ImportError: If ``transformers`` is not installed.
        """
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for ASRTransformers. "
                "Install it with: pip install graph_ragu[local]"
            ) from exc

        options: dict[str, Any] = {
            "model": self.model_name_or_path,
            "chunk_length_s": self.chunk_length_s,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            **self.pipeline_kwargs,
        }
        if self.device is not None:
            options["device"] = self.device

        self._pipeline = pipeline("automatic-speech-recognition", **options)

    def _transcribe_sync(
        self,
        audio: str | Path | bytes,
        language: str | None,
        prompt: str | None = None,
    ) -> Any:
        """
        Run blocking inference.

        :param audio: Path to an audio file, or its encoded bytes.
        :param language: Language hint.
        :param prompt: Decoder prompt.
        :return: Raw pipeline output.
        """
        return self._pipeline(
            _pipeline_payload(audio),
            return_timestamps=self.return_timestamps,
            generate_kwargs=self._generation_options(language, prompt),
        )

    def _transcribe_many_sync(
        self,
        payloads: List[str | bytes],
        language: str | None,
        prompt: str | None = None,
    ) -> List[Any]:
        """
        Run blocking inference over a whole batch.

        ``batch_size`` has to be passed to the call, not just a list of inputs:
        a pipeline handed a list still walks it one item at a time unless told
        the batch width, so without this the batch would only look like one.

        :param payloads: Pipeline inputs, already normalized.
        :type payloads: List[str | bytes]
        :param language: Language hint.
        :type language: str | None
        :param prompt: Decoder prompt.
        :type prompt: str | None
        :returns: One raw pipeline output per input, in order.
        :rtype: List[Any]
        """
        outputs = self._pipeline(
            payloads,
            return_timestamps=self.return_timestamps,
            generate_kwargs=self._generation_options(language, prompt),
            batch_size=len(payloads),
        )

        if isinstance(outputs, (dict, str)):
            return [outputs]
        return list(outputs)

    def _generation_options(
        self,
        language: str | None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Build the ``generate_kwargs`` for one call.

        :param language: Per-call language hint; falls back to the instance
            default.
        :param prompt: Decoder prompt; falls back to the instance default.
        :return: Generation options, possibly empty.
        """
        options = dict(self.generate_kwargs)

        effective_language = language if language is not None else self.language
        if effective_language:
            options["language"] = effective_language
        if self.task:
            options.setdefault("task", self.task)

        effective_prompt = prompt if prompt is not None else self.prompt
        prompt_ids = self._prompt_ids(effective_prompt)
        if prompt_ids is not None:
            options.setdefault("prompt_ids", prompt_ids)

        return options

    @staticmethod
    async def _duration(audio: str | Path | bytes) -> float | None:
        """
        Probe the recording length, used to close open-ended timestamps.

        :param audio: Path to an audio file, or its encoded bytes.
        :returns: Duration in seconds, or ``None`` if unavailable.
        :rtype: float | None
        """
        if isinstance(audio, bytes):
            return None
        try:
            return await probe_duration(audio)
        except RuntimeError:
            # ffprobe is optional here: without it, open-ended timestamps are
            # clamped to their own start instead of the real end of the file.
            return None

    def _to_transcript(self, output: Any, duration: float | None) -> Transcript:
        """
        Map pipeline output onto a :class:`Transcript`.

        With ``return_timestamps="word"`` the pipeline emits one chunk per word
        and no segmentation at all, so the words are grouped back into segments
        here. With ``return_timestamps=True`` the chunks are already segments
        and carry no words.

        :param output: Raw pipeline output.
        :param duration: Recording length, when known.
        :returns: Parsed transcript.
        :rtype: Transcript
        """
        if isinstance(output, str):
            return Transcript(text=output.strip(), model=self.model_name_or_path)

        spans = self._read_chunks(output, duration)

        if self.return_timestamps == "word":
            words = [
                TranscriptWord(text=text, start=start, end=end)
                for start, end, text in spans
            ]
            segments = group_words_into_segments(words)
        else:
            segments = [
                TranscriptSegment(start=start, end=end, text=text)
                for start, end, text in spans
            ]

        return Transcript(
            text=(output.get("text") or "").strip(),
            segments=segments,
            language=self.language,
            duration=duration,
            model=self.model_name_or_path,
        )

    @staticmethod
    def _read_chunks(
        output: dict[str, Any],
        duration: float | None,
    ) -> List[tuple[float, float, str]]:
        """
        Read the pipeline's timed chunks, whatever they represent.

        :param output: Raw pipeline output.
        :type output: dict[str, Any]
        :param duration: Recording length, used to close an open-ended chunk.
        :type duration: float | None
        :returns: ``(start, end, text)`` triples in timeline order.
        :rtype: List[tuple[float, float, str]]
        """
        spans: list[tuple[float, float, str]] = []

        for chunk in output.get("chunks") or ():
            timestamp = chunk.get("timestamp") or (None, None)
            start, end = timestamp[0], timestamp[1]
            if start is None:
                continue
            # In long-form decoding the trailing chunk may have an open end.
            if end is None:
                end = duration if duration is not None else start
            spans.append((float(start), float(end), (chunk.get("text") or "").strip()))

        return spans
