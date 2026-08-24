"""Unit tests: the local ASR backend and the helpers that rebuild its output.

`ASRTransformers` gets a stubbed pipeline rather than a real model — what is
under test is the layer around it: what the pipeline is asked for, and how its
output is turned back into a `Transcript`. That output needs more work than the
hosted one: asked for word timestamps the pipeline reports one chunk per word
and no segmentation at all, and it strips the spacing the words came with.
"""
import pytest

from ragu.models.asr import (
    ASRTransformers,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    group_words_into_segments,
    join_word_texts,
)


def word(text: str, start: float, end: float) -> TranscriptWord:
    return TranscriptWord(text=text, start=start, end=end)


class TestJoinWordTexts:
    """Backends disagree about whether a token carries its own spacing."""

    def test_tokens_carrying_their_own_space_are_concatenated(self):
        # faster-whisper and the OpenAI API hand back " Редактор", ".Кулак" —
        # concatenating verbatim reproduces the original text exactly.
        words = [word(" Привет", 0.0, 1.0), word(" мир", 1.0, 2.0)]

        assert join_word_texts(words) == "Привет мир"

    def test_bare_tokens_are_rejoined_with_spaces(self):
        words = [word("Привет", 0.0, 1.0), word("мир", 1.0, 2.0)]

        assert join_word_texts(words) == "Привет мир"

    def test_clinging_punctuation_does_not_take_a_space(self):
        # Naively joining bare tokens with spaces produces "А . Кулак".
        words = [word("Привет", 0.0, 1.0), word(",", 1.0, 1.1), word("мир", 1.1, 2.0)]

        assert join_word_texts(words) == "Привет, мир"

    def test_an_opening_bracket_clings_to_what_follows(self):
        words = [word("см", 0.0, 1.0), word("(", 1.0, 1.1), word("рис", 1.1, 2.0)]

        assert join_word_texts(words) == "см (рис"

    def test_a_single_leading_space_switches_the_whole_run(self):
        # One spaced token means the backend spaces them all; mixing strategies
        # per word would double the spaces it already provided.
        words = [word(" Привет", 0.0, 1.0), word("мир", 1.0, 2.0)]

        assert join_word_texts(words) == "Приветмир"

    def test_no_words_is_an_empty_string(self):
        assert join_word_texts([]) == ""

    def test_blank_tokens_are_skipped(self):
        assert join_word_texts([word("Привет", 0.0, 1.0), word("   ", 1.0, 1.1)]) == "Привет"


class TestGroupWordsIntoSegments:
    """Everything downstream works in segments; word mode reports none."""

    def test_a_pause_starts_a_new_segment(self):
        words = [word("раз", 0.0, 0.5), word("два", 2.0, 2.5)]

        segments = group_words_into_segments(words, max_gap=0.6)

        assert [s.text for s in segments] == ["раз", "два"]

    def test_a_short_gap_stays_in_one_segment(self):
        words = [word("раз", 0.0, 0.5), word("два", 0.7, 1.2)]

        segments = group_words_into_segments(words, max_gap=0.6)

        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 1.2

    def test_the_duration_cap_cuts_a_long_run(self):
        words = [word(f"w{i}", float(i), float(i) + 0.5) for i in range(10)]

        segments = group_words_into_segments(words, max_gap=5.0, max_duration=4.0)

        assert len(segments) > 1
        assert all(s.end - s.start <= 4.0 for s in segments)

    def test_a_sentence_ending_cuts_once_the_segment_is_worth_cutting(self):
        words = [
            word("Привет", 0.0, 0.5),
            word("мир.", 0.6, 1.5),
            word("Дальше", 1.6, 2.2),
        ]

        segments = group_words_into_segments(words, max_gap=5.0)

        assert [s.text for s in segments] == ["Привет мир.", "Дальше"]

    def test_a_sentence_ending_does_not_cut_a_segment_barely_started(self):
        # Cutting on every abbreviation and initial would shred the transcript.
        words = [word("см.", 0.0, 0.3), word("рис", 0.4, 0.8)]

        segments = group_words_into_segments(words, max_gap=5.0)

        assert len(segments) == 1

    def test_segments_carry_their_words(self):
        words = [word("раз", 0.0, 0.5), word("два", 2.0, 2.5)]

        segments = group_words_into_segments(words, max_gap=0.6)

        assert [len(s.words) for s in segments] == [1, 1]

    def test_no_words_is_no_segments(self):
        assert group_words_into_segments([]) == []


class StubPipeline:
    """Stands in for the transformers ASR pipeline, recording its arguments."""

    def __init__(self, output) -> None:
        self.output = output
        self.calls: list[dict] = []

    def __call__(self, payload, **kwargs):
        self.calls.append({"payload": payload, **kwargs})
        return self.output


@pytest.fixture
def stubbed_asr(monkeypatch):
    """ASRTransformers with the pipeline replaced and no duration probe.

    ffprobe is stubbed to a fixed length rather than skipped, so the tests do
    not depend on ffmpeg being installed.
    """
    def build(output, duration: float | None = 10.0, **kwargs) -> ASRTransformers:
        asr = ASRTransformers(**kwargs)
        asr._pipeline = StubPipeline(output)

        async def fake_duration(audio):
            return duration

        monkeypatch.setattr(ASRTransformers, "_duration", staticmethod(fake_duration))
        return asr

    return build


class FakeTensor:
    """Stands in for the tensor `get_prompt_ids` returns."""

    def __init__(self, text: str) -> None:
        self.text = text

    def to(self, _device):
        return self


class BatchStubPipeline:
    """Stands in for the pipeline when it is handed a whole batch.

    Returns one output per input, so a caller that quietly dropped or reordered
    a batch member is caught rather than papered over.
    """

    def __init__(self, failing: set[str] | None = None, max_width: int | None = None) -> None:
        self.failing = failing or set()
        self.max_width = max_width
        self.calls: list[list] = []
        self.widths: list[int | None] = []
        self.generate_kwargs: list[dict] = []

    def __call__(self, payload, **kwargs):
        batch = payload if isinstance(payload, list) else [payload]
        self.calls.append(list(batch))
        self.widths.append(kwargs.get("batch_size"))
        self.generate_kwargs.append(dict(kwargs.get("generate_kwargs") or {}))

        # A batch too wide for the card fails as a whole while every recording
        # in it is perfectly readable — the shape of a CUDA OOM.
        if self.max_width is not None and len(batch) > self.max_width:
            raise RuntimeError(f"out of memory for a batch of {len(batch)}")

        refused = self.failing.intersection(batch)
        if refused:
            raise RuntimeError(f"cannot decode {sorted(refused)}")

        outputs = [{"text": f"text-{item}", "chunks": []} for item in batch]
        return outputs if isinstance(payload, list) else outputs[0]


@pytest.fixture
def batched_asr(monkeypatch):
    """ASRTransformers whose pipeline reports what batches it was handed."""
    def build(failing: set[str] | None = None, max_width: int | None = None, **kwargs) -> tuple:
        asr = ASRTransformers(**kwargs)
        pipeline = BatchStubPipeline(failing, max_width)
        asr._pipeline = pipeline

        async def fake_duration(audio):
            return 10.0

        monkeypatch.setattr(ASRTransformers, "_duration", staticmethod(fake_duration))
        return asr, pipeline

    return build


class TestLocalTrueBatching:
    """One pipeline call per batch, not one per recording.

    The base implementation starts `batch_size` coroutines that the inference
    semaphore then serializes — the same work in sequence, for a `batch_size`
    that bought nothing.
    """

    async def test_one_pipeline_call_per_batch(self, batched_asr):
        asr, pipeline = batched_asr()

        await asr.transcribe_batch(["a.wav", "b.wav", "c.wav", "d.wav"], batch_size=2)

        assert len(pipeline.calls) == 2
        assert pipeline.calls[0] == ["a.wav", "b.wav"]
        assert pipeline.calls[1] == ["c.wav", "d.wav"]

    async def test_results_keep_input_order(self, batched_asr):
        asr, _pipeline = batched_asr()

        results = await asr.transcribe_batch(["a.wav", "b.wav", "c.wav"], batch_size=2)

        assert [r.text for r in results] == ["text-a.wav", "text-b.wav", "text-c.wav"]

    async def test_a_single_batch_holds_everything_when_it_fits(self, batched_asr):
        asr, pipeline = batched_asr()

        await asr.transcribe_batch(["a.wav", "b.wav"], batch_size=8)

        assert len(pipeline.calls) == 1

    async def test_the_batch_width_is_passed_alongside_the_list(self, batched_asr):
        # A pipeline handed a list still walks it one item at a time unless it
        # is told the width, so without this the batch only looks like one.
        asr, pipeline = batched_asr()

        await asr.transcribe_batch(["a.wav", "b.wav", "c.wav"], batch_size=2)

        assert pipeline.widths == [2, 1]

    async def test_paths_reach_the_pipeline_as_strings(self, batched_asr, tmp_path):
        asr, pipeline = batched_asr()
        path = tmp_path / "clip.wav"

        await asr.transcribe_batch([path], batch_size=1)

        assert pipeline.calls[0] == [str(path)]

    async def test_the_language_hint_reaches_the_batch(self, batched_asr):
        asr, pipeline = batched_asr(language="ru")

        await asr.transcribe_batch(["a.wav"], batch_size=1)

        assert asr._generation_options(None)["language"] == "ru"

    async def test_the_duration_is_attached_to_every_transcript(self, batched_asr):
        asr, _pipeline = batched_asr()

        results = await asr.transcribe_batch(["a.wav", "b.wav"], batch_size=2)

        assert [r.duration for r in results] == [10.0, 10.0]

    async def test_batch_size_must_be_positive(self, batched_asr):
        asr, _pipeline = batched_asr()

        with pytest.raises(ValueError):
            await asr.transcribe_batch(["a.wav"], batch_size=0)

    async def test_inference_is_serialized_across_batches(self, batched_asr):
        asr, _pipeline = batched_asr(max_concurrency=1)
        active = peak = 0

        original = asr._transcribe_many_sync

        def counting(payloads, language, prompt=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                return original(payloads, language, prompt)
            finally:
                active -= 1

        asr._transcribe_many_sync = counting

        await asr.transcribe_batch(["a.wav", "b.wav", "c.wav", "d.wav"], batch_size=2)

        assert peak == 1


class TestLocalBatchFailures:
    """A batch that fails as a whole must not cost every recording in it."""

    async def test_a_failed_batch_is_retried_one_at_a_time(self, batched_asr):
        asr, pipeline = batched_asr(failing={"bad.wav"})

        results = await asr.transcribe_batch(
            ["good.wav", "bad.wav"], batch_size=2, continue_on_error=True
        )

        assert results[0] is not None
        assert results[0].text == "text-good.wav"
        assert results[1] is None

    async def test_the_retry_only_touches_the_failed_batch(self, batched_asr):
        asr, pipeline = batched_asr(failing={"bad.wav"})

        await asr.transcribe_batch(
            ["a.wav", "b.wav", "bad.wav", "c.wav"], batch_size=2, continue_on_error=True
        )

        # First batch as a batch; second batch once as a batch, then singly.
        assert pipeline.calls[0] == ["a.wav", "b.wav"]
        assert pipeline.calls[1] == ["bad.wav", "c.wav"]
        assert ["bad.wav"] in pipeline.calls
        assert ["c.wav"] in pipeline.calls

    async def test_a_good_recording_grouped_with_a_bad_one_survives(self, batched_asr):
        asr, _pipeline = batched_asr(failing={"bad.wav"})

        results = await asr.transcribe_batch(
            ["bad.wav", "fine.wav"], batch_size=2, continue_on_error=True
        )

        assert results[0] is None
        assert results[1].text == "text-fine.wav"

    async def test_the_default_still_raises(self, batched_asr):
        asr, _pipeline = batched_asr(failing={"bad.wav"})

        with pytest.raises(RuntimeError):
            await asr.transcribe_batch(["good.wav", "bad.wav"], batch_size=2)

    async def test_a_batch_too_wide_recovers_every_recording(self, batched_asr):
        # The CUDA-OOM shape: the batch fails as a whole, but each recording is
        # fine on its own, so retrying singly should lose nothing at all.
        asr, _pipeline = batched_asr(max_width=2)

        results = await asr.transcribe_batch(
            ["a.wav", "b.wav", "c.wav", "d.wav"], batch_size=4, continue_on_error=True
        )

        assert [r.text for r in results] == [
            "text-a.wav", "text-b.wav", "text-c.wav", "text-d.wav"
        ]

    async def test_the_retry_runs_outside_the_exception_handler(self, batched_asr):
        # Retrying inside the handler keeps the failed batch's tensors alive
        # through the exception's traceback, so an out-of-memory batch fails
        # again one recording at a time. `sys.exc_info` is how that shows up
        # without a GPU: it is empty once the handler has ended.
        import sys

        asr, _pipeline = batched_asr(max_width=1)
        seen: list = []

        original = asr._transcribe_singly

        async def watching(batch, language, prompt, progress):
            seen.append(sys.exc_info()[0])
            return await original(batch, language, prompt, progress)

        asr._transcribe_singly = watching

        await asr.transcribe_batch(["a.wav", "b.wav"], batch_size=2, continue_on_error=True)

        assert seen == [None]


class TestSerialization:
    """`to_dict` and `from_dict` have to be each other's inverse.

    Parsers put segments into document metadata; anything reading them back
    should get objects rather than having to walk raw dictionaries itself.
    """

    @staticmethod
    def _transcript() -> Transcript:
        return Transcript(
            text="Привет, мир.",
            segments=[
                TranscriptSegment(
                    start=0.0,
                    end=3.0,
                    text="Привет, мир.",
                    speaker="SPEAKER_00",
                    words=[
                        TranscriptWord(text=" Привет,", start=0.0, end=1.4, confidence=0.9),
                        TranscriptWord(text=" мир.", start=1.5, end=3.0, speaker="SPEAKER_00"),
                    ],
                )
            ],
            language="ru",
            duration=3.0,
            model="whisper-1",
        )

    def test_a_transcript_survives_the_round_trip(self):
        original = self._transcript()

        assert Transcript.from_dict(original.to_dict()) == original

    def test_a_segment_without_words_survives(self):
        segment = TranscriptSegment(start=1.0, end=2.0, text="без слов")

        assert TranscriptSegment.from_dict(segment.to_dict()) == segment

    def test_a_transcript_without_segments_survives(self):
        transcript = Transcript(text="no timestamps", model="gpt-4o-transcribe")

        assert Transcript.from_dict(transcript.to_dict()) == transcript

    def test_stored_segment_mappings_read_back_as_objects(self):
        # This is the shape a parser writes under metadata["segments"].
        stored = [segment.to_dict() for segment in self._transcript().segments]

        rebuilt = [TranscriptSegment.from_dict(item) for item in stored]

        assert rebuilt == self._transcript().segments
        assert rebuilt[0].words[0].confidence == 0.9

    def test_missing_optional_fields_are_tolerated(self):
        # A hand-written or older mapping carries only the bounds.
        segment = TranscriptSegment.from_dict({"start": 0, "end": 1})

        assert segment.text == ""
        assert segment.speaker is None
        assert segment.words == []


class TestLocalTranscription:
    async def test_a_bare_string_output_becomes_a_transcript(self, stubbed_asr):
        # A pipeline asked for no timestamps returns the text and nothing else.
        asr = stubbed_asr("  Привет мир  ", return_timestamps=False)

        transcript = await asr.transcribe(b"audio")

        assert transcript.text == "Привет мир"
        assert transcript.segments == []

    async def test_timestamped_chunks_become_segments(self, stubbed_asr):
        output = {
            "text": "Привет мир",
            "chunks": [
                {"timestamp": (0.0, 1.4), "text": " Привет"},
                {"timestamp": (1.5, 3.0), "text": " мир"},
            ],
        }
        asr = stubbed_asr(output, return_timestamps=True)

        transcript = await asr.transcribe(b"audio")

        assert [(s.start, s.end, s.text) for s in transcript.segments] == [
            (0.0, 1.4, "Привет"),
            (1.5, 3.0, "мир"),
        ]
        # Segment mode reports no words, which is what downgrades speaker
        # assignment to whole-segment overlap.
        assert all(s.words == [] for s in transcript.segments)

    async def test_word_mode_regroups_words_into_segments(self, stubbed_asr):
        # In word mode the pipeline emits one chunk per word and no segments.
        output = {
            "text": "раз два",
            "chunks": [
                {"timestamp": (0.0, 0.5), "text": "раз"},
                {"timestamp": (3.0, 3.5), "text": "два"},
            ],
        }
        asr = stubbed_asr(output, return_timestamps="word")

        transcript = await asr.transcribe(b"audio")

        assert [s.text for s in transcript.segments] == ["раз", "два"]
        assert [w.text for s in transcript.segments for w in s.words] == ["раз", "два"]

    async def test_an_open_ended_final_chunk_is_closed_with_the_duration(
        self, stubbed_asr,
    ):
        # Long-form decoding leaves the trailing chunk's end unset.
        output = {"text": "хвост", "chunks": [{"timestamp": (8.0, None), "text": "хвост"}]}
        asr = stubbed_asr(output, duration=12.5, return_timestamps=True)

        transcript = await asr.transcribe(b"audio")

        assert transcript.segments[0].end == 12.5

    async def test_an_open_ended_chunk_without_a_duration_collapses(self, stubbed_asr):
        # Without ffprobe there is nothing better to close it with.
        output = {"text": "хвост", "chunks": [{"timestamp": (8.0, None), "text": "хвост"}]}
        asr = stubbed_asr(output, duration=None, return_timestamps=True)

        transcript = await asr.transcribe(b"audio")

        assert transcript.segments[0].end == 8.0

    async def test_a_chunk_without_a_start_is_dropped(self, stubbed_asr):
        output = {
            "text": "Привет",
            "chunks": [
                {"timestamp": (None, None), "text": "?"},
                {"timestamp": (0.0, 1.0), "text": "Привет"},
            ],
        }
        asr = stubbed_asr(output, return_timestamps=True)

        transcript = await asr.transcribe(b"audio")

        assert [s.text for s in transcript.segments] == ["Привет"]

    async def test_the_transcript_reports_the_model_and_duration(self, stubbed_asr):
        asr = stubbed_asr(
            {"text": "x", "chunks": []},
            duration=42.0,
            model_name_or_path="openai/whisper-small",
            language="ru",
        )

        transcript = await asr.transcribe(b"audio")

        assert transcript.model == "openai/whisper-small"
        assert transcript.duration == 42.0
        assert transcript.language == "ru"


class TestPromptSupport:
    """The prompt is applied where the backend can take it, reported where not.

    The pipeline has no `prompt` of its own, but Whisper's `generate` accepts
    `prompt_ids` and the pipeline forwards `generate_kwargs` to it.
    """

    class WhisperishTokenizer:
        """A tokenizer that can build prompt tokens, like Whisper's."""

        def __init__(self):
            self.calls: list[str] = []

        def get_prompt_ids(self, prompt, return_tensors=None):
            self.calls.append(prompt)
            return FakeTensor(prompt)

    class PlainTokenizer:
        """A CTC tokenizer: no decoder to condition, so no prompt tokens."""

    @staticmethod
    def _asr(tokenizer, **kwargs):
        asr = ASRTransformers(**kwargs)
        pipeline = BatchStubPipeline()
        pipeline.tokenizer = tokenizer
        pipeline.model = type("M", (), {"device": "cpu"})()
        asr._pipeline = pipeline
        return asr, pipeline

    async def test_the_prompt_reaches_generate_as_prompt_ids(self):
        tokenizer = self.WhisperishTokenizer()
        asr, pipeline = self._asr(tokenizer)

        await asr.transcribe_batch(["a.wav"], batch_size=1, desc=None, prompt="РАГУ, NEREL")

        assert tokenizer.calls == ["РАГУ, NEREL"]
        assert pipeline.generate_kwargs[0]["prompt_ids"].text == "РАГУ, NEREL"

    async def test_no_prompt_means_no_prompt_ids(self):
        asr, pipeline = self._asr(self.WhisperishTokenizer())

        await asr.transcribe_batch(["a.wav"], batch_size=1, desc=None)

        assert "prompt_ids" not in pipeline.generate_kwargs[0]

    async def test_an_instance_default_prompt_is_used(self):
        tokenizer = self.WhisperishTokenizer()
        asr, _pipeline = self._asr(tokenizer, prompt="default terms")

        await asr.transcribe_batch(["a.wav"], batch_size=1, desc=None)

        assert tokenizer.calls == ["default terms"]

    async def test_a_per_call_prompt_wins_over_the_default(self):
        tokenizer = self.WhisperishTokenizer()
        asr, _pipeline = self._asr(tokenizer, prompt="default")

        await asr.transcribe_batch(["a.wav"], batch_size=1, desc=None, prompt="per call")

        assert tokenizer.calls == ["per call"]

    async def test_explicit_prompt_ids_are_not_overwritten(self):
        # A caller who built their own meant them.
        tokenizer = self.WhisperishTokenizer()
        asr, pipeline = self._asr(
            tokenizer, generate_kwargs={"prompt_ids": FakeTensor("mine")}
        )

        await asr.transcribe_batch(["a.wav"], batch_size=1, desc=None, prompt="ours")

        assert pipeline.generate_kwargs[0]["prompt_ids"].text == "mine"

    async def test_the_tokens_are_built_once_per_prompt(self):
        tokenizer = self.WhisperishTokenizer()
        asr, _pipeline = self._asr(tokenizer)

        await asr.transcribe_batch(
            ["a.wav", "b.wav", "c.wav"], batch_size=1, desc=None, prompt="terms"
        )

        assert tokenizer.calls == ["terms"]

    async def test_a_backend_without_prompt_support_reports_once(self):
        from ragu.common.logger import logger

        asr, pipeline = self._asr(self.PlainTokenizer())
        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(message), level="WARNING")
        try:
            await asr.transcribe_batch(
                ["a.wav", "b.wav"], batch_size=1, desc=None, prompt="terms"
            )
        finally:
            logger.remove(sink)

        assert len([w for w in warnings if "cannot apply a decoder prompt" in w]) == 1
        assert all("prompt_ids" not in gk for gk in pipeline.generate_kwargs)

    async def test_a_backend_without_prompt_support_stays_quiet_without_one(self):
        from ragu.common.logger import logger

        asr, _pipeline = self._asr(self.PlainTokenizer())
        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(message), level="WARNING")
        try:
            await asr.transcribe_batch(["a.wav"], batch_size=1, desc=None)
        finally:
            logger.remove(sink)

        assert not [w for w in warnings if "cannot apply a decoder prompt" in w]


class TestLocalGenerationOptions:
    async def test_the_language_hint_reaches_generate(self, stubbed_asr):
        asr = stubbed_asr({"text": "x", "chunks": []}, language="ru")

        await asr.transcribe(b"audio")

        assert asr._pipeline.calls[0]["generate_kwargs"]["language"] == "ru"

    async def test_a_per_call_hint_wins_over_the_default(self, stubbed_asr):
        asr = stubbed_asr({"text": "x", "chunks": []}, language="ru")

        await asr.transcribe(b"audio", language="en")

        assert asr._pipeline.calls[0]["generate_kwargs"]["language"] == "en"

    async def test_the_task_is_passed_through(self, stubbed_asr):
        asr = stubbed_asr({"text": "x", "chunks": []}, task="translate")

        await asr.transcribe(b"audio")

        assert asr._pipeline.calls[0]["generate_kwargs"]["task"] == "translate"

    async def test_a_path_is_handed_over_as_a_string(self, stubbed_asr, tmp_path):
        # The pipeline reads the file itself; handing it bytes would decode twice.
        path = tmp_path / "talk.wav"
        path.write_bytes(b"audio")
        asr = stubbed_asr({"text": "x", "chunks": []})

        await asr.transcribe(path)

        assert asr._pipeline.calls[0]["payload"] == str(path)

    async def test_an_ignored_prompt_is_reported_once(self, stubbed_asr):
        from ragu.common.logger import logger

        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(message), level="WARNING")
        try:
            asr = stubbed_asr({"text": "x", "chunks": []})
            await asr.transcribe(b"audio", prompt="термины")
            await asr.transcribe(b"audio", prompt="термины")
        finally:
            logger.remove(sink)

        assert len([w for w in warnings if "prompt" in w]) == 1

    async def test_no_prompt_is_no_warning(self, stubbed_asr):
        from ragu.common.logger import logger

        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(message), level="WARNING")
        try:
            await stubbed_asr({"text": "x", "chunks": []}).transcribe(b"audio")
        finally:
            logger.remove(sink)

        assert warnings == []

    @pytest.mark.parametrize("bad", [0, -1])
    async def test_concurrency_must_be_positive(self, bad):
        with pytest.raises(ValueError, match="max_concurrency must be positive"):
            ASRTransformers(max_concurrency=bad)
