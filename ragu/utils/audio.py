"""
ffmpeg-backed audio helpers shared by ASR clients and audio/video parsers.

Everything here shells out to ``ffmpeg``/``ffprobe`` instead of adding a
Python audio dependency: the binaries are needed anyway for anything beyond
plain WAV, and transformers' ASR pipeline already relies on them.
"""

import asyncio
import glob as globlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

FFMPEG_INSTALL_HINT = (
    "ffmpeg is required for audio/video processing. "
    "Install it with your package manager, e.g. `apt install ffmpeg` "
    "or `brew install ffmpeg`."
)


def ffmpeg_available() -> bool:
    """
    Check whether the ``ffmpeg`` binary is on PATH.

    :returns: ``True`` if ffmpeg can be executed.
    :rtype: bool
    """
    return shutil.which("ffmpeg") is not None


def require_ffmpeg() -> None:
    """
    Ensure ffmpeg is installed.

    :raises RuntimeError: If the ``ffmpeg`` binary is not on PATH.
    """
    if not ffmpeg_available():
        raise RuntimeError(FFMPEG_INSTALL_HINT)


async def _run(program: str, *args: str) -> str:
    """
    Run a binary and return its stdout.

    :param program: Executable name.
    :param args: Command-line arguments.
    :returns: Captured stdout, decoded as UTF-8.
    :rtype: str
    :raises RuntimeError: If the binary is missing or exits non-zero.
    """
    if shutil.which(program) is None:
        raise RuntimeError(FFMPEG_INSTALL_HINT)

    process = await asyncio.create_subprocess_exec(
        program,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip().splitlines()
        tail = message[-1] if message else f"exit code {process.returncode}"
        raise RuntimeError(f"{program} failed: {tail}")

    return stdout.decode("utf-8", errors="replace")


async def probe_duration(path: str | Path) -> float | None:
    """
    Read the duration of a media file in seconds.

    :param path: Path to an audio or video file.
    :type path: str | Path
    :returns: Duration in seconds, or ``None`` if the container does not
        report one.
    :rtype: float | None
    :raises RuntimeError: If ``ffprobe`` is missing or fails.
    """
    output = await _run(
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    )
    try:
        return float(output.strip())
    except ValueError:
        return None


async def extract_audio(
    path: str | Path,
    destination: str | Path,
    sample_rate: int = 16000,
) -> Path:
    """
    Extract a mono audio track, resampled for speech models.

    Works for both audio and video containers, so video parsing is just this
    call followed by regular audio transcription.

    :param path: Path to the source media file.
    :type path: str | Path
    :param destination: Path of the resulting file; the suffix picks the codec
        (``.wav`` for PCM, ``.mp3`` for a much smaller upload).
    :type destination: str | Path
    :param sample_rate: Target sample rate in Hz.
    :type sample_rate: int
    :returns: Path to the extracted audio.
    :rtype: Path
    :raises RuntimeError: If ffmpeg is missing or fails.
    """
    destination = Path(destination)
    await _run(
        "ffmpeg",
        "-y",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        str(destination),
    )
    return destination


@dataclass(slots=True)
class AudioPreprocess:
    """
    How to bring a recording to the canonical form the models expect.

    Applied in one ffmpeg pass, which is the point: speech-to-text and
    diarization otherwise decode the same file twice, each with its own
    resampler, and neither of them normalizes loudness or forces a predictable
    channel layout.

    :param sample_rate: Target sample rate in Hz. 16 kHz is what every speech
        model here consumes; anything higher is discarded downstream.
    :param channels: Target channel count. Mono matters for diarization — a
        stereo interview with the speakers panned apart otherwise depends on
        which channel the backend happens to pick.
    :param normalize: Loudness normalization. ``"ebu"`` applies EBU R128
        (``loudnorm``), ``"dynamic"`` applies ``dynaudnorm`` for recordings
        whose level drifts, ``None`` leaves the level alone.
    :param target_lufs: Integrated loudness target for ``"ebu"``, in LUFS.
        ``-16`` is the usual target for speech; broadcast uses ``-23``.
    :param highpass_hz: Corner frequency of a high-pass filter removing rumble,
        handling noise and HVAC. ``None`` disables it.
    :param denoise: Broadband denoiser: ``"afftdn"`` (spectral) or ``"anlmdn"``
        (non-local means). Off by default on purpose — on already-clean
        recordings denoising reliably costs accuracy rather than adding it.
    :param extra_filters: Extra ffmpeg filter expressions appended to the
        chain, for anything this config does not cover.
    """

    sample_rate: int = 16000
    channels: int = 1
    normalize: Literal["ebu", "dynamic"] | None = "ebu"
    target_lufs: float = -16.0
    highpass_hz: int | None = 80
    denoise: Literal["afftdn", "anlmdn"] | None = None
    extra_filters: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels < 1:
            raise ValueError(f"channels must be at least 1, got {self.channels}")
        if self.highpass_hz is not None and self.highpass_hz <= 0:
            raise ValueError(f"highpass_hz must be positive, got {self.highpass_hz}")


def build_filter_chain(config: AudioPreprocess) -> str:
    """
    Render an :class:`AudioPreprocess` as an ffmpeg ``-af`` expression.

    Order is deliberate: rumble is removed first so the denoiser and the
    loudness meter are not chasing energy nobody wants, and normalization comes
    last so it measures the signal that actually survives.

    :param config: Preprocessing settings.
    :type config: AudioPreprocess
    :returns: Filter chain, empty when nothing but resampling was asked for.
    :rtype: str
    """
    filters: list[str] = []

    if config.highpass_hz is not None:
        filters.append(f"highpass=f={config.highpass_hz}")

    if config.denoise == "afftdn":
        filters.append("afftdn=nf=-25")
    elif config.denoise == "anlmdn":
        filters.append("anlmdn")

    if config.normalize == "ebu":
        # Single-pass loudnorm is less exact than the two-pass form, but it
        # costs one decode instead of two and the residual error is well under
        # what speech models care about.
        filters.append(f"loudnorm=I={config.target_lufs}:TP=-1.5:LRA=11")
    elif config.normalize == "dynamic":
        filters.append("dynaudnorm=f=250:g=15")

    filters.extend(config.extra_filters)
    return ",".join(filters)


async def preprocess_audio(
    path: str | Path,
    destination: str | Path,
    config: AudioPreprocess | None = None,
) -> Path:
    """
    Convert a recording to canonical form in a single ffmpeg pass.

    Works for video containers too — the video stream is dropped — so callers
    handling both need no separate extraction step.

    The output is 16-bit PCM WAV: reading it back costs almost nothing compared
    with decoding the source again, which is what makes it worth handing the
    same file to several models.

    :param path: Path to the source audio or video file.
    :type path: str | Path
    :param destination: Path of the resulting WAV file.
    :type destination: str | Path
    :param config: Preprocessing settings; defaults to :class:`AudioPreprocess`.
    :type config: AudioPreprocess | None
    :returns: Path to the preprocessed audio.
    :rtype: Path
    :raises RuntimeError: If ffmpeg is missing or fails.
    """
    settings = config or AudioPreprocess()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)

    chain = build_filter_chain(settings)

    arguments = [
        "-nostdin",
        "-y",
        "-i", str(path),
        "-vn",
        *(("-af", chain) if chain else ()),
        # Cover art rides along as a video stream in many podcast files and
        # confuses the muxer once the audio is re-encoded.
        "-map_metadata", "-1",
        "-ac", str(settings.channels),
        "-ar", str(settings.sample_rate),
        "-c:a", "pcm_s16le",
        str(output),
    ]

    await _run("ffmpeg", *arguments)
    return output


async def split_audio(
    path: str | Path,
    chunk_seconds: float,
    destination_dir: str | Path,
    sample_rate: int = 16000,
) -> List[Tuple[Path, float]]:
    """
    Split audio into fixed-length parts for providers with upload limits.

    Callers shift transcript timestamps by each part's offset to rebuild a
    single timeline, so the offsets have to be the real ones. The segment muxer
    cuts at the first frame at or after each boundary rather than exactly on it,
    which puts a part of a second between `index * chunk_seconds` and where a
    part actually starts — small per part, but it accumulates down a long
    recording. ffmpeg is therefore asked to report the cuts it made, and the
    even spacing is only a fallback for when it does not.

    The parts come from that report rather than from a directory listing, so a
    file name carrying glob metacharacters cannot make them disappear.

    :param path: Path to the source audio file.
    :type path: str | Path
    :param chunk_seconds: Length of each part in seconds.
    :type chunk_seconds: float
    :param destination_dir: Directory to write the parts into.
    :type destination_dir: str | Path
    :param sample_rate: Target sample rate in Hz.
    :type sample_rate: int
    :returns: ``(path, offset_seconds)`` pairs, in timeline order.
    :rtype: List[Tuple[Path, float]]
    :raises RuntimeError: If ffmpeg is missing or fails.
    :raises ValueError: If `chunk_seconds` is not positive.
    """
    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds must be positive, got {chunk_seconds}")

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    pattern = destination_dir / f"{stem}-%05d.mp3"
    listing = destination_dir / "segments.csv"

    await _run(
        "ffmpeg",
        "-y",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-segment_list", str(listing),
        "-segment_list_type", "csv",
        str(pattern),
    )

    reported = _read_segment_offsets(listing)
    if reported:
        return [
            (destination_dir / name, offset)
            for name, offset in sorted(reported.items(), key=lambda item: (item[1], item[0]))
        ]

    # ffmpeg wrote no readable segment list. Finding the parts on disk is the
    # fallback, and the stem has to be escaped before it goes into a glob
    # pattern: a recording named `lecture[1].mp3` otherwise matches nothing and
    # the caller is handed an empty split of a recording that was cut fine.
    parts = sorted(destination_dir.glob(f"{globlib.escape(stem)}-*.mp3"))
    return [(part, index * chunk_seconds) for index, part in enumerate(parts)]


def _read_segment_offsets(listing: Path) -> dict[str, float]:
    """
    Read the start offset ffmpeg recorded for each part it wrote.

    The segment list is ``name,start,end`` per line. Anything unreadable is
    skipped rather than raised on: the audio was split successfully either way,
    and the caller falls back to even spacing for the parts left unaccounted.

    :param listing: Path of the segment list ffmpeg was asked to write.
    :type listing: Path
    :returns: Start offset in seconds, keyed by part file name.
    :rtype: dict[str, float]
    """
    if not listing.exists():
        return {}

    offsets: dict[str, float] = {}
    for line in listing.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(",")
        if len(fields) < 2:
            continue
        try:
            offsets[Path(fields[0]).name] = float(fields[1])
        except ValueError:
            continue

    return offsets
