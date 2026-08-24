"""Unit tests: the hosted ASR and OCR backends against a stubbed model client.

These stub `CachedAsyncOpenAI` rather than the SDK underneath it — that seam is
covered by `tests/llm/test_transcribe.py`. What is under test here is the layer
above: which request each backend builds for a given model, and how it turns a
provider payload back into RAGU's own types.
"""
import asyncio
from pathlib import Path

import httpx
import pytest
from openai import BadRequestError

from ragu.models.asr import ASROpenAI
from ragu.models.ocr import (
    DEFAULT_OCR_PROMPT,
    TABLE_RECOGNITION_PROMPT,
    OCROpenAI,
)

VERBOSE_PAYLOAD = {
    "text": "Привет мир",
    "language": "russian",
    "duration": 3.0,
    "segments": [
        {"start": 0.0, "end": 1.4, "text": " Привет"},
        {"start": 1.5, "end": 3.0, "text": " мир"},
    ],
    "words": [
        {"word": " Привет", "start": 0.0, "end": 1.4},
        {"word": " мир", "start": 1.5, "end": 3.0},
    ],
}


def _bad_request(message: str) -> BadRequestError:
    """Build the 400 an endpoint returns for a format it does not support."""
    return BadRequestError(
        message,
        response=httpx.Response(400, request=httpx.Request("POST", "http://stub/v1")),
        body=None,
    )


class FakeClient:
    """Stands in for CachedAsyncOpenAI, recording what it was asked for.

    `rejects` names the response formats this endpoint answers with a 400,
    which is what a model served under a name the format table does not know
    does when asked for `verbose_json`.
    """

    def __init__(
        self,
        payload=None,
        response: str | None = "recognized text",
        rejects: tuple[str, ...] = (),
    ) -> None:
        self.payload = payload if payload is not None else VERBOSE_PAYLOAD
        self.response = response
        self.rejects = rejects
        self.calls: list[dict] = []

    async def transcribe(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if kwargs.get("response_format") in self.rejects:
            raise _bad_request(f"unsupported response_format {kwargs['response_format']}")
        payload = self.payload
        return payload(kwargs) if callable(payload) else payload

    async def chat_completion(self, **kwargs) -> str | None:
        self.calls.append(kwargs)
        return self.response


class TestResponseFormat:
    async def test_whisper_gets_timestamps_and_words(self):
        client = FakeClient()

        await ASROpenAI(client=client).transcribe(b"audio")

        call = client.calls[0]
        assert call["response_format"] == "verbose_json"
        assert call["timestamp_granularities"] == ["segment", "word"]

    async def test_a_diarizing_model_asks_for_speakers(self):
        client = FakeClient()

        await ASROpenAI(client=client, model_name="gpt-4o-transcribe-diarize").transcribe(
            b"audio"
        )

        assert client.calls[0]["response_format"] == "diarized_json"

    async def test_a_model_without_timestamps_falls_back_to_plain_json(self):
        client = FakeClient(payload={"text": "no timestamps"})

        await ASROpenAI(client=client, model_name="gpt-4o-transcribe").transcribe(b"audio")

        call = client.calls[0]
        assert call["response_format"] == "json"
        # Granularities are meaningless without verbose_json and some providers
        # reject the combination outright.
        assert "timestamp_granularities" not in call

    async def test_the_timestamp_warning_is_not_repeated(self):
        from ragu.common.logger import logger

        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(message), level="WARNING")
        try:
            asr = ASROpenAI(client=FakeClient(payload={"text": ""}),
                            model_name="gpt-4o-transcribe")
            await asr.transcribe(b"audio")
            await asr.transcribe(b"audio")
        finally:
            logger.remove(sink)

        assert len([w for w in warnings if "timestamps" in w]) == 1

    async def test_an_explicit_format_is_passed_through(self):
        client = FakeClient(payload={"text": "srt body"})

        await ASROpenAI(client=client, response_format="srt").transcribe(b"audio")

        assert client.calls[0]["response_format"] == "srt"


class TestFormatDowngrade:
    """The format table holds OpenAI's names; endpoints use their own.

    An Azure deployment or a vLLM server names its models whatever it likes, so
    a model that cannot answer `verbose_json` gets asked for it anyway. The 400
    that comes back must not reach the caller as a failed transcription.
    """

    async def test_a_rejected_format_is_retried_as_plain_json(self):
        client = FakeClient(payload={"text": "recognized"}, rejects=("verbose_json",))

        transcript = await ASROpenAI(client=client, model_name="my-whisper").transcribe(
            b"audio"
        )

        assert [call["response_format"] for call in client.calls] == ["verbose_json", "json"]
        assert transcript.text == "recognized"
        assert transcript.segments == []

    async def test_granularities_are_dropped_with_the_format(self):
        client = FakeClient(payload={"text": "recognized"}, rejects=("verbose_json",))

        await ASROpenAI(client=client, model_name="my-whisper").transcribe(b"audio")

        # Asking for word timings alongside plain json is the same 400 again.
        assert "timestamp_granularities" not in client.calls[1]

    async def test_the_downgrade_sticks_for_later_recordings(self):
        client = FakeClient(payload={"text": "recognized"}, rejects=("verbose_json",))

        asr = ASROpenAI(client=client, model_name="my-whisper")
        await asr.transcribe(b"first")
        await asr.transcribe(b"second")

        # One rejection, not one per recording.
        assert [call["response_format"] for call in client.calls] == [
            "verbose_json", "json", "json",
        ]

    async def test_the_downgrade_is_announced_once(self):
        from ragu.common.logger import logger

        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(message), level="WARNING")
        try:
            asr = ASROpenAI(
                client=FakeClient(payload={"text": ""}, rejects=("verbose_json",)),
                model_name="my-whisper",
            )
            await asr.transcribe(b"first")
            await asr.transcribe(b"second")
        finally:
            logger.remove(sink)

        assert len([w for w in warnings if "falling back to plain 'json'" in w]) == 1

    async def test_a_diarizing_format_downgrades_too(self):
        client = FakeClient(payload={"text": "recognized"}, rejects=("diarized_json",))

        await ASROpenAI(
            client=client, model_name="gpt-4o-transcribe-diarize",
        ).transcribe(b"audio")

        assert [call["response_format"] for call in client.calls] == ["diarized_json", "json"]

    async def test_an_explicit_format_is_not_second_guessed(self):
        # The caller asked for this one; overriding it would hide their mistake.
        client = FakeClient(rejects=("verbose_json",))

        with pytest.raises(BadRequestError):
            await ASROpenAI(client=client, response_format="verbose_json").transcribe(b"audio")

        assert len(client.calls) == 1

    async def test_a_rejection_that_is_not_about_the_format_still_raises(self):
        client = FakeClient(rejects=("verbose_json", "json"))

        with pytest.raises(BadRequestError):
            await ASROpenAI(client=client, model_name="my-whisper").transcribe(b"audio")

        # Downgraded once, then gave up rather than looping.
        assert len(client.calls) == 2

    async def test_a_rejection_about_another_parameter_is_not_downgraded(self):
        # The provider names the parameter it refused. An unsupported language
        # is not a format this adapter guessed wrong, and retrying it in plain
        # json would send the same rejected language again.
        class PickyClient(FakeClient):
            async def transcribe(self, **kwargs):
                self.calls.append(kwargs)
                raise BadRequestError(
                    "Unsupported language",
                    response=httpx.Response(
                        400, request=httpx.Request("POST", "http://stub/v1")
                    ),
                    body={"param": "language"},
                )

        client = PickyClient()

        with pytest.raises(BadRequestError):
            await ASROpenAI(client=client, model_name="my-whisper").transcribe(b"audio")

        assert len(client.calls) == 1

    async def test_a_rejection_about_the_granularities_is_downgraded(self):
        # Word timings and the rich format stand or fall together.
        class GranularityClient(FakeClient):
            async def transcribe(self, **kwargs):
                self.calls.append(kwargs)
                if "timestamp_granularities" in kwargs:
                    raise BadRequestError(
                        "not supported",
                        response=httpx.Response(
                            400, request=httpx.Request("POST", "http://stub/v1")
                        ),
                        body={"param": "timestamp_granularities"},
                    )
                return {"text": "recognized"}

        client = GranularityClient()

        transcript = await ASROpenAI(
            client=client, model_name="my-whisper",
        ).transcribe(b"audio")

        assert transcript.text == "recognized"
        assert [call["response_format"] for call in client.calls] == ["verbose_json", "json"]

    async def test_a_server_that_names_only_the_format_is_understood(self):
        # vLLM answers `Currently do not support diarized_json for <model>` with
        # param unset: it names the format, never the field it arrived in.
        class VLLMClient(FakeClient):
            async def transcribe(self, **kwargs):
                self.calls.append(kwargs)
                fmt = kwargs.get("response_format")
                if fmt != "json":
                    raise BadRequestError(
                        f"Currently do not support {fmt} for podlodka_turbo",
                        response=httpx.Response(
                            400, request=httpx.Request("POST", "http://stub/v1")
                        ),
                        body={"param": None},
                    )
                return {"text": "recognized"}

        client = VLLMClient()

        transcript = await ASROpenAI(
            client=client, model_name="podlodka_turbo",
        ).transcribe(b"audio")

        assert transcript.text == "recognized"
        assert [call["response_format"] for call in client.calls] == ["verbose_json", "json"]

    async def test_an_undecodable_upload_is_not_mistaken_for_a_format(self):
        # The same server answers this for audio it cannot read. Nothing about
        # the format is wrong, so retrying in a poorer one would only repeat it.
        class UnreadableClient(FakeClient):
            async def transcribe(self, **kwargs):
                self.calls.append(kwargs)
                raise BadRequestError(
                    "Invalid or unsupported audio file.",
                    response=httpx.Response(
                        400, request=httpx.Request("POST", "http://stub/v1")
                    ),
                    body={"param": None},
                )

        client = UnreadableClient()

        with pytest.raises(BadRequestError, match="audio file"):
            await ASROpenAI(client=client, model_name="podlodka_turbo").transcribe(b"audio")

        assert len(client.calls) == 1

    async def test_a_model_already_on_plain_json_does_not_repeat_itself(self):
        # Nothing richer was asked for, so there is no poorer format to fall
        # back to and the retry would be the identical request.
        client = FakeClient(rejects=("json",))

        with pytest.raises(BadRequestError):
            await ASROpenAI(
                client=client, model_name="gpt-4o-transcribe",
            ).transcribe(b"audio")

        assert len(client.calls) == 1

    async def test_words_asked_for_but_not_returned_are_reported(self):
        # vLLM answers `"words": null` beside good segments: the request is
        # accepted, the words simply never come, and speaker assignment silently
        # drops to whole-segment overlap.
        from ragu.common.logger import logger

        payload = {
            "text": "Привет мир",
            "duration": 3.0,
            "segments": [{"start": 0.0, "end": 3.0, "text": "Привет мир"}],
            "words": None,
        }
        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(message), level="WARNING")
        try:
            asr = ASROpenAI(client=FakeClient(payload=payload), model_name="my-whisper")
            await asr.transcribe(b"first")
            await asr.transcribe(b"second")
        finally:
            logger.remove(sink)

        reported = [w for w in warnings if "returned none" in w]
        assert len(reported) == 1  # once per instance, not once per recording

    async def test_no_word_warning_when_words_arrive(self):
        from ragu.common.logger import logger

        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(message), level="WARNING")
        try:
            await ASROpenAI(client=FakeClient(), model_name="my-whisper").transcribe(b"a")
        finally:
            logger.remove(sink)

        assert not [w for w in warnings if "returned none" in w]

    async def test_no_word_warning_when_words_were_not_asked_for(self):
        from ragu.common.logger import logger

        payload = {
            "text": "Привет",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Привет"}],
            "words": None,
        }
        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(message), level="WARNING")
        try:
            await ASROpenAI(
                client=FakeClient(payload=payload),
                model_name="my-whisper",
                timestamp_granularities=("segment",),
            ).transcribe(b"a")
        finally:
            logger.remove(sink)

        assert not [w for w in warnings if "returned none" in w]

    async def test_granularities_can_be_narrowed_to_segments(self):
        client = FakeClient()

        await ASROpenAI(client=client, timestamp_granularities=("segment",)).transcribe(
            b"audio"
        )

        assert client.calls[0]["timestamp_granularities"] == ["segment"]


class TestRequestOptions:
    async def test_instance_defaults_are_sent(self):
        client = FakeClient()

        await ASROpenAI(client=client, language="ru", prompt="RAGU").transcribe(b"audio")

        call = client.calls[0]
        assert call["language"] == "ru"
        assert call["prompt"] == "RAGU"

    async def test_a_per_call_hint_wins_over_the_default(self):
        client = FakeClient()

        await ASROpenAI(client=client, language="ru").transcribe(b"audio", language="en")

        assert client.calls[0]["language"] == "en"

    async def test_unset_options_are_omitted_entirely(self):
        client = FakeClient()

        await ASROpenAI(client=client).transcribe(b"audio")

        assert "language" not in client.calls[0]
        assert "prompt" not in client.calls[0]

    async def test_extra_options_reach_the_client(self):
        client = FakeClient()

        await ASROpenAI(client=client, temperature=0.0).transcribe(b"audio")

        assert client.calls[0]["temperature"] == 0.0

    async def test_a_path_is_read_and_named(self, tmp_path):
        path = tmp_path / "lecture.mp3"
        path.write_bytes(b"audio-bytes")
        client = FakeClient()

        await ASROpenAI(client=client).transcribe(path)

        call = client.calls[0]
        assert call["audio"] == b"audio-bytes"
        # The provider reads the container format off the name.
        assert call["audio_name"] == "lecture.mp3"


class TestPayloadParsing:
    async def test_segments_carry_their_words(self):
        transcript = await ASROpenAI(client=FakeClient()).transcribe(b"audio")

        assert transcript.text == "Привет мир"
        assert transcript.language == "russian"
        assert transcript.duration == 3.0
        assert transcript.model == "whisper-1"
        assert [segment.text for segment in transcript.segments] == ["Привет", "мир"]
        assert [w.text for w in transcript.segments[0].words] == [" Привет"]
        assert [w.text for w in transcript.segments[1].words] == [" мир"]

    async def test_a_word_is_placed_by_its_midpoint(self):
        # The word runs 0.9-1.6, straddling the 1.0 boundary; most of it lies in
        # the second segment, so that is where it belongs.
        payload = {
            "text": "a b",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "a"},
                {"start": 1.0, "end": 2.0, "text": "b"},
            ],
            "words": [{"word": "straddler", "start": 0.9, "end": 1.6}],
        }

        transcript = await ASROpenAI(client=FakeClient(payload)).transcribe(b"audio")

        assert transcript.segments[0].words == []
        assert [w.text for w in transcript.segments[1].words] == ["straddler"]

    async def test_segments_without_bounds_are_dropped(self):
        payload = {
            "text": "a b",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "kept"},
                {"start": None, "end": None, "text": "dropped"},
            ],
        }

        transcript = await ASROpenAI(client=FakeClient(payload)).transcribe(b"audio")

        assert [segment.text for segment in transcript.segments] == ["kept"]

    async def test_speaker_labels_survive_from_a_diarizing_model(self):
        payload = {
            "text": "hello there",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A"},
                {"start": 1.0, "end": 2.0, "text": "there", "speaker": "B"},
            ],
        }

        transcript = await ASROpenAI(
            client=FakeClient(payload), model_name="gpt-4o-transcribe-diarize"
        ).transcribe(b"audio")

        assert [segment.speaker for segment in transcript.segments] == ["A", "B"]

    async def test_a_payload_without_timestamps_is_text_only(self):
        transcript = await ASROpenAI(
            client=FakeClient({"text": "no timestamps"}),
            model_name="gpt-4o-transcribe",
        ).transcribe(b"audio")

        assert transcript.text == "no timestamps"
        assert transcript.segments == []
        assert transcript.duration is None


@pytest.fixture(params=["bytes", "path"])
def oversized(request, tmp_path):
    """The same oversized recording, handed over in each accepted input shape.

    Both shapes have to reach the splitter: they take different routes through
    it — bytes are written out for ffmpeg, a path is handed to ffmpeg directly —
    and only one of them used to be exercised.
    """
    data = b"a long recording"
    if request.param == "bytes":
        return data

    path = tmp_path / "lecture.wav"
    path.write_bytes(data)
    return path


class TestSplitting:
    """Recordings above the upload limit are cut up and stitched back together."""

    @pytest.fixture
    def split_into_two(self, monkeypatch):
        async def fake_split(path, chunk_seconds, destination_dir, sample_rate=16000):
            destination_dir = Path(destination_dir)
            destination_dir.mkdir(parents=True, exist_ok=True)
            parts = []
            for index in range(2):
                part = destination_dir / f"part-{index}.mp3"
                part.write_bytes(f"part-{index}".encode())
                parts.append((part, index * chunk_seconds))
            return parts

        monkeypatch.setattr("ragu.models.asr.split_audio", fake_split)

    @staticmethod
    def _payload_per_part(kwargs) -> dict:
        index = int(kwargs["audio"].decode().removeprefix("part-"))
        return {
            "text": f"text-{index}",
            "language": "ru",
            "segments": [{"start": 0.0, "end": 1.0, "text": f"text-{index}"}],
        }

    async def test_a_small_recording_is_sent_whole(self, split_into_two):
        client = FakeClient()

        await ASROpenAI(client=client, max_upload_bytes=1024).transcribe(b"tiny")

        assert len(client.calls) == 1
        assert client.calls[0]["audio"] == b"tiny"

    async def test_parts_are_shifted_onto_one_timeline(self, split_into_two, oversized):
        client = FakeClient(self._payload_per_part)

        transcript = await ASROpenAI(
            client=client, max_upload_bytes=8, split_seconds=600.0,
        ).transcribe(oversized)

        assert len(client.calls) == 2
        # The second part's timestamps are offset by its position, not left
        # starting from zero again.
        assert [(s.start, s.end) for s in transcript.segments] == [
            (0.0, 1.0), (600.0, 601.0),
        ]

    async def test_the_stitched_transcript_reads_as_one(self, split_into_two, oversized):
        transcript = await ASROpenAI(
            client=FakeClient(self._payload_per_part),
            max_upload_bytes=8,
            split_seconds=600.0,
        ).transcribe(oversized)

        assert transcript.text == "text-0 text-1"
        assert transcript.language == "ru"
        # No provider reports a duration for the whole file, so it comes from
        # how far the last segment reaches.
        assert transcript.duration == 601.0

    async def test_parts_are_uploaded_a_few_at_a_time(self, monkeypatch, oversized):
        # Eighteen parts of a three-hour recording in flight at once is how a
        # hosted backend hits its rate limit.
        async def fake_split(path, chunk_seconds, destination_dir, sample_rate=16000):
            destination_dir = Path(destination_dir)
            destination_dir.mkdir(parents=True, exist_ok=True)
            parts = []
            for index in range(6):
                part = destination_dir / f"part-{index}.mp3"
                part.write_bytes(f"part-{index}".encode())
                parts.append((part, index * chunk_seconds))
            return parts

        monkeypatch.setattr("ragu.models.asr.split_audio", fake_split)

        active = peak = 0

        class CountingClient(FakeClient):
            async def transcribe(self, **kwargs):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0)
                    return {"text": "part", "segments": []}
                finally:
                    active -= 1

        client = CountingClient()
        await ASROpenAI(
            client=client, max_upload_bytes=8, max_parallel_parts=2,
        ).transcribe(oversized)

        assert len(client.calls) == 0  # overridden transcribe records nothing
        assert peak == 2

    @pytest.mark.parametrize("bad", [0, -1])
    async def test_the_part_limit_must_be_positive(self, bad):
        with pytest.raises(ValueError, match="max_parallel_parts must be positive"):
            ASROpenAI(client=FakeClient(), max_parallel_parts=bad)

    async def test_a_split_that_produced_nothing_is_loud(self, monkeypatch, oversized):
        # An empty split stitches into an empty transcript, which is
        # indistinguishable from a silent recording.
        async def fake_split(path, chunk_seconds, destination_dir, sample_rate=16000):
            Path(destination_dir).mkdir(parents=True, exist_ok=True)
            return []

        monkeypatch.setattr("ragu.models.asr.split_audio", fake_split)

        with pytest.raises(RuntimeError, match="no parts"):
            await ASROpenAI(
                client=FakeClient(self._payload_per_part), max_upload_bytes=8,
            ).transcribe(oversized)

    async def test_parts_still_over_the_limit_are_refused(self, split_into_two, oversized):
        # Splitting is decided in bytes but performed in seconds, so a
        # split_seconds too generous for the bitrate yields parts the provider
        # would reject. Naming the setting beats relaying a 400.
        with pytest.raises(ValueError, match="split_seconds"):
            await ASROpenAI(
                client=FakeClient(self._payload_per_part), max_upload_bytes=4,
            ).transcribe(oversized)

    async def test_in_memory_audio_is_written_out_before_splitting(self, monkeypatch):
        # Splitting shells out to ffmpeg, which needs a file even when the
        # caller handed over bytes.
        # Read inside the stub: the temporary directory is gone by the time
        # `transcribe` returns.
        written: list[bytes] = []

        async def fake_split(path, chunk_seconds, destination_dir, sample_rate=16000):
            written.append(Path(path).read_bytes())
            part = Path(destination_dir)
            part.mkdir(parents=True, exist_ok=True)
            (part / "part-0.mp3").write_bytes(b"part-0")
            return [(part / "part-0.mp3", 0.0)]

        monkeypatch.setattr("ragu.models.asr.split_audio", fake_split)

        await ASROpenAI(
            client=FakeClient(self._payload_per_part), max_upload_bytes=8,
        ).transcribe(b"a long recording")

        assert written == [b"a long recording"]

    async def test_a_file_is_split_in_place_and_never_read_whole(self, monkeypatch, tmp_path):
        # The recording is split precisely because it is too big to upload, so
        # loading it into memory to find that out defeats the exercise: ffmpeg
        # reads it from disk and only the parts come back into memory.
        source = tmp_path / "lecture.wav"
        source.write_bytes(b"a long recording")

        seen: list[Path] = []

        async def fake_split(path, chunk_seconds, destination_dir, sample_rate=16000):
            seen.append(Path(path))
            part = Path(destination_dir)
            part.mkdir(parents=True, exist_ok=True)
            (part / "part-0.mp3").write_bytes(b"part-0")
            return [(part / "part-0.mp3", 0.0)]

        monkeypatch.setattr("ragu.models.asr.split_audio", fake_split)

        read = Path.read_bytes
        loaded: list[Path] = []

        def tracking_read(self):
            loaded.append(self)
            return read(self)

        monkeypatch.setattr(Path, "read_bytes", tracking_read)

        await ASROpenAI(
            client=FakeClient(self._payload_per_part), max_upload_bytes=8,
        ).transcribe(source)

        assert seen == [source]
        assert source not in loaded


class TestOCR:
    async def test_the_image_is_sent_as_a_data_uri(self):
        client = FakeClient()

        await OCROpenAI(client=client).recognize(b"\x89PNG-bytes")

        content = client.calls[0]["conversation"][0]["content"]
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["text"] == DEFAULT_OCR_PROMPT

    async def test_the_mime_type_follows_the_image(self):
        client = FakeClient()

        await OCROpenAI(client=client).recognize(b"jpeg-bytes", mime="image/jpeg")

        url = client.calls[0]["conversation"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    async def test_a_per_call_prompt_wins(self):
        client = FakeClient()

        await OCROpenAI(client=client).recognize(
            b"png", prompt=TABLE_RECOGNITION_PROMPT
        )

        content = client.calls[0]["conversation"][0]["content"]
        assert content[1]["text"] == TABLE_RECOGNITION_PROMPT

    async def test_generation_options_reach_the_client(self):
        client = FakeClient()

        await OCROpenAI(client=client, model_name="glm", temperature=0.0).recognize(b"png")

        call = client.calls[0]
        assert call["model_name"] == "glm"
        assert call["temperature"] == 0.0

    async def test_the_recognized_text_is_stripped(self):
        client = FakeClient(response="  # Heading\n\ntext  \n")

        result = await OCROpenAI(client=client).recognize(b"png")

        assert result == "# Heading\n\ntext"

    async def test_an_empty_response_is_an_empty_string(self):
        # A page the model found nothing on must not become the string "None".
        result = await OCROpenAI(client=FakeClient(response=None)).recognize(b"png")

        assert result == ""

    async def test_a_batch_keeps_input_order(self):
        client = FakeClient()
        client.response = "page"

        results = await OCROpenAI(client=client).recognize_batch(
            [b"a", b"b", b"c"], batch_size=2
        )

        assert results == ["page", "page", "page"]
        assert len(client.calls) == 3
