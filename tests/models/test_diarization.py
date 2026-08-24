"""
Tests for the diarization model type and the speaker merge.

The pyannote backend is exercised only through its guard rails — the checkpoint
is gated and multi-gigabyte, so anything past model loading belongs in an
integration run, not here. The merge, which is where the actual logic lives, is
covered directly.
"""
from __future__ import annotations

import pytest

from ragu.models.asr import (
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    join_word_texts,
)
from ragu.models.diarization import (
    Diarization,
    DiarizationPyannote,
    SpeakerTurn,
    _token_kwargs,
    assign_speakers,
    merge_adjacent,
    overlap,
    parse_pyannote_output,
)


def word(text: str, start: float, end: float) -> TranscriptWord:
    return TranscriptWord(text=text, start=start, end=end)


@pytest.fixture
def dialogue():
    """
    Two ASR segments, the second one straddling a change of speaker.

    This is the case the whole word-overlap path exists for: a recognizer cuts
    on pauses, so "...спасибо" and "а у тебя" land in one segment even though
    two people said them.
    """
    return Transcript(
        text="Привет как дела Всё отлично спасибо а у тебя",
        segments=[
            TranscriptSegment(
                start=0.0, end=2.0, text="Привет как дела",
                words=[word("Привет", 0.0, 0.6), word("как", 0.7, 1.2), word("дела", 1.3, 2.0)],
            ),
            TranscriptSegment(
                start=2.1, end=4.5, text="Всё отлично спасибо а у тебя",
                words=[
                    word("Всё", 2.1, 2.5), word("отлично", 2.6, 3.1), word("спасибо", 3.2, 3.6),
                    word("а", 3.9, 4.0), word("у", 4.1, 4.2), word("тебя", 4.3, 4.5),
                ],
            ),
        ],
        duration=5.0,
        model="fake-whisper",
    )


@pytest.fixture
def turns():
    return [
        SpeakerTurn(0.0, 2.05, "SPEAKER_00"),
        SpeakerTurn(2.05, 3.8, "SPEAKER_01"),
        SpeakerTurn(3.8, 4.6, "SPEAKER_00"),
    ]


class TestOverlap:
    @pytest.mark.parametrize(
        "spans,expected",
        [
            ((0.0, 2.0, 1.0, 3.0), 1.0),
            ((0.0, 2.0, 2.0, 3.0), 0.0),
            ((0.0, 5.0, 1.0, 2.0), 1.0),
            ((3.0, 4.0, 0.0, 1.0), 0.0),
        ],
    )
    def test_intersection_length(self, spans, expected):
        assert overlap(*spans) == expected


class TestWordOverlap:
    def test_a_segment_spanning_a_turn_boundary_is_split(self, dialogue, turns):
        result = assign_speakers(dialogue, turns)

        assert [(s.speaker, s.text) for s in result.segments] == [
            ("SPEAKER_00", "Привет как дела"),
            ("SPEAKER_01", "Всё отлично спасибо"),
            ("SPEAKER_00", "а у тебя"),
        ]

    def test_every_segment_ends_up_with_exactly_one_speaker(self, dialogue, turns):
        result = assign_speakers(dialogue, turns)

        for segment in result.segments:
            speakers = {w.speaker for w in segment.words}
            assert len(speakers) == 1
            assert segment.speaker in speakers

    def test_split_bounds_follow_the_words(self, dialogue, turns):
        result = assign_speakers(dialogue, turns)

        assert result.segments[1].end == 3.6
        assert result.segments[2].start == 3.9

    def test_the_input_transcript_is_not_modified(self, dialogue, turns):
        assign_speakers(dialogue, turns)

        assert len(dialogue.segments) == 2
        assert all(segment.speaker is None for segment in dialogue.segments)

    def test_transcript_level_fields_are_carried_over(self, dialogue, turns):
        result = assign_speakers(dialogue, turns)

        assert result.language == dialogue.language
        assert result.duration == 5.0
        assert result.model == "fake-whisper"


class TestSegmentOverlap:
    def test_segments_are_labelled_but_not_split(self, dialogue, turns):
        result = assign_speakers(dialogue, turns, strategy="segment_overlap")

        assert len(result.segments) == 2
        assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]

    def test_it_misattributes_the_straddling_tail(self, dialogue, turns):
        # Documents the known cost of having no word timings: "а у тебя" was
        # said by SPEAKER_00 but is kept under SPEAKER_01.
        result = assign_speakers(dialogue, turns, strategy="segment_overlap")

        assert "а у тебя" in result.segments[1].text
        assert result.segments[1].speaker == "SPEAKER_01"

    def test_a_transcript_without_words_downgrades_automatically(self, turns):
        transcript = Transcript(
            text="весь диалог одним куском",
            segments=[TranscriptSegment(start=0.0, end=4.5, text="весь диалог одним куском")],
        )

        result = assign_speakers(transcript, turns)

        assert len(result.segments) == 1
        assert result.segments[0].speaker == "SPEAKER_00"


class TestEdgeCases:
    def test_no_turns_yields_an_unlabelled_copy(self, dialogue):
        # The result is always the caller's to edit: returning the argument
        # itself here would make it an alias on exactly the recordings where
        # the diarizer found nobody, and a copy everywhere else.
        result = assign_speakers(dialogue, [])

        assert result is not dialogue
        assert result.segments[0] is not dialogue.segments[0]
        assert all(segment.speaker is None for segment in result.segments)
        assert [s.text for s in result.segments] == [s.text for s in dialogue.segments]

    def test_no_segments_yields_an_empty_copy(self, turns):
        transcript = Transcript(text="плоский текст")

        result = assign_speakers(transcript, turns)

        assert result is not transcript
        assert result.segments == []
        assert result.text == "плоский текст"

    def test_editing_the_result_does_not_reach_the_input(self, dialogue):
        result = assign_speakers(dialogue, [])

        result.segments[0].text = "переписано"

        assert dialogue.segments[0].text != "переписано"

    def test_segments_that_already_have_a_speaker_pass_through(self, turns):
        transcript = Transcript(
            text="x",
            segments=[TranscriptSegment(start=0.0, end=4.5, text="x", speaker="Alice")],
        )

        result = assign_speakers(transcript, turns)

        assert [s.speaker for s in result.segments] == ["Alice"]

    def test_a_word_in_a_gap_between_turns_takes_the_nearest_speaker(self):
        transcript = Transcript(
            text="эхо",
            segments=[
                TranscriptSegment(start=10.0, end=10.5, text="эхо", words=[word("эхо", 10.0, 10.5)])
            ],
        )
        far_turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00"), SpeakerTurn(11.0, 12.0, "SPEAKER_01")]

        result = assign_speakers(transcript, far_turns)

        assert result.segments[0].speaker == "SPEAKER_01"

    def test_an_unknown_strategy_is_rejected(self, dialogue, turns):
        with pytest.raises(ValueError, match="Unknown strategy"):
            assign_speakers(dialogue, turns, strategy="vibes")

    def test_a_long_turn_is_found_behind_a_later_short_one(self):
        transcript = Transcript(
            text="слово",
            segments=[
                TranscriptSegment(start=50.0, end=51.0, text="слово",
                                  words=[word("слово", 50.0, 51.0)])
            ],
        )
        # The backward scan must reach past the shorter, later-starting turn.
        overlapping = [SpeakerTurn(0.0, 100.0, "LONG"), SpeakerTurn(10.0, 10.1, "SHORT")]

        result = assign_speakers(transcript, overlapping)

        assert result.segments[0].speaker == "LONG"


class TestMergeAdjacent:
    def test_the_input_segments_are_not_grown_in_place(self):
        first = TranscriptSegment(0.0, 1.0, "Привет", speaker="A")
        second = TranscriptSegment(1.2, 2.0, "как дела", speaker="A")

        merge_adjacent([first, second], max_gap=0.5)

        assert (first.text, first.end) == ("Привет", 1.0)
        assert second.text == "как дела"

    def test_same_speaker_runs_are_rejoined(self):
        merged = merge_adjacent(
            [
                TranscriptSegment(0.0, 1.0, "Привет", speaker="A"),
                TranscriptSegment(1.2, 2.0, "как дела", speaker="A"),
            ],
            max_gap=0.5,
        )

        assert len(merged) == 1
        assert merged[0].text == "Привет как дела"
        assert merged[0].end == 2.0

    def test_a_speaker_change_blocks_the_merge(self):
        merged = merge_adjacent(
            [
                TranscriptSegment(0.0, 1.0, "a", speaker="A"),
                TranscriptSegment(1.1, 2.0, "b", speaker="B"),
            ],
            max_gap=0.5,
        )

        assert len(merged) == 2

    def test_a_long_gap_blocks_the_merge(self):
        merged = merge_adjacent(
            [
                TranscriptSegment(0.0, 1.0, "a", speaker="A"),
                TranscriptSegment(9.0, 10.0, "b", speaker="A"),
            ],
            max_gap=0.5,
        )

        assert len(merged) == 2

    def test_the_duration_cap_blocks_the_merge(self):
        merged = merge_adjacent(
            [
                TranscriptSegment(0.0, 8.0, "a", speaker="A"),
                TranscriptSegment(8.1, 12.0, "b", speaker="A"),
            ],
            max_gap=0.5,
            max_duration=10.0,
        )

        assert len(merged) == 2


class Segment:
    """Stand-in for ``pyannote.core.Segment``."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class Annotation3x:
    """pyannote.audio 3.x: the pipeline returns the Annotation itself."""

    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for index, (segment, label) in enumerate(self._tracks):
            yield (segment, f"_{index}", label) if yield_label else (segment, f"_{index}")


class Annotation4x:
    """pyannote.audio 4.x: iterating the annotation yields (segment, label)."""

    def __init__(self, tracks):
        self._tracks = tracks

    def __iter__(self):
        return iter(self._tracks)


class DiarizeOutput:
    """pyannote.audio 4.x wraps the annotation in a result object."""

    def __init__(self, annotation):
        self.speaker_diarization = annotation
        self.embeddings = None


class TestMergedTextMatchesWords:
    """A merged segment's text has to be what its words say.

    Joining two texts with a space puts one where punctuation belongs, which is
    the whole reason `join_word_texts` exists.
    """

    def test_merging_rebuilds_the_text_from_the_words(self):
        first = TranscriptSegment(
            start=0.0, end=1.0, text="А", speaker="S0",
            words=[TranscriptWord(text="А", start=0.0, end=1.0, speaker="S0")],
        )
        second = TranscriptSegment(
            start=1.1, end=2.0, text=".Кулак", speaker="S0",
            words=[TranscriptWord(text=".Кулак", start=1.1, end=2.0, speaker="S0")],
        )

        merged = merge_adjacent([first, second])

        assert len(merged) == 1
        assert merged[0].text == join_word_texts(merged[0].words)
        assert merged[0].text == "А.Кулак"

    def test_every_merged_segment_agrees_with_its_words(self):
        words = [
            TranscriptWord(text="Раз", start=0.0, end=0.5, speaker="S0"),
            TranscriptWord(text=",", start=0.5, end=0.6, speaker="S0"),
            TranscriptWord(text="два", start=0.7, end=1.2, speaker="S0"),
            TranscriptWord(text="!", start=1.2, end=1.3, speaker="S0"),
        ]
        segments = [
            TranscriptSegment(
                start=word.start, end=word.end, text=word.text,
                speaker="S0", words=[word],
            )
            for word in words
        ]

        for segment in merge_adjacent(segments):
            assert segment.text == join_word_texts(segment.words)

    def test_segments_without_words_still_join_with_a_space(self):
        first = TranscriptSegment(start=0.0, end=1.0, text="Раз", speaker="S0")
        second = TranscriptSegment(start=1.1, end=2.0, text="два", speaker="S0")

        merged = merge_adjacent([first, second])

        assert merged[0].text == "Раз два"

    def test_a_side_without_words_does_not_lose_its_text(self):
        # Rebuilding from one side's words alone would drop the other side.
        first = TranscriptSegment(
            start=0.0, end=1.0, text="Раз", speaker="S0",
            words=[TranscriptWord(text="Раз", start=0.0, end=1.0, speaker="S0")],
        )
        second = TranscriptSegment(start=1.1, end=2.0, text="два", speaker="S0")

        merged = merge_adjacent([first, second])

        assert "Раз" in merged[0].text
        assert "два" in merged[0].text


class TestPyannoteOutputParsing:
    """
    Both pyannote generations must be readable.

    These shapes are reproduced from the errors 4.x actually produced against
    this adapter: ``'DiarizeOutput' object has no attribute 'itertracks'``.
    """

    tracks = [
        (Segment(0.0, 2.0), "SPEAKER_00"),
        (Segment(2.0, 4.0), "SPEAKER_01"),
    ]

    def test_3x_annotation(self):
        turns = parse_pyannote_output(Annotation3x(self.tracks))

        assert [(t.start, t.end, t.speaker) for t in turns] == [
            (0.0, 2.0, "SPEAKER_00"),
            (2.0, 4.0, "SPEAKER_01"),
        ]

    def test_4x_diarize_output(self):
        turns = parse_pyannote_output(DiarizeOutput(Annotation4x(self.tracks)))

        assert [(t.start, t.end, t.speaker) for t in turns] == [
            (0.0, 2.0, "SPEAKER_00"),
            (2.0, 4.0, "SPEAKER_01"),
        ]

    def test_4x_wrapper_around_an_annotation_that_still_has_itertracks(self):
        turns = parse_pyannote_output(DiarizeOutput(Annotation3x(self.tracks)))

        assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"]

    def test_a_bare_4x_annotation_without_the_wrapper(self):
        turns = parse_pyannote_output(Annotation4x(self.tracks))

        assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"]

    def test_an_empty_result_is_no_turns(self):
        assert parse_pyannote_output(DiarizeOutput(Annotation4x([]))) == []

    def test_an_unreadable_object_is_reported_clearly(self):
        with pytest.raises(RuntimeError, match="neither an Annotation nor iterable"):
            parse_pyannote_output(object())

    def test_segments_without_labels_are_reported_clearly(self):
        with pytest.raises(RuntimeError, match="without speaker labels"):
            parse_pyannote_output(Annotation4x([Segment(0.0, 1.0)]))


class TestTokenKeyword:
    """
    The auth argument was renamed in 4.x and the old spelling is now rejected:
    ``from_pretrained() got an unexpected keyword argument 'use_auth_token'``.
    """

    def test_4x_uses_token(self):
        class Pipeline4x:
            @staticmethod
            def from_pretrained(checkpoint, token=None, **kwargs):
                ...

        assert _token_kwargs(Pipeline4x, "hf_x") == {"token": "hf_x"}

    def test_3x_uses_use_auth_token(self):
        class Pipeline3x:
            @staticmethod
            def from_pretrained(checkpoint, use_auth_token=None, **kwargs):
                ...

        assert _token_kwargs(Pipeline3x, "hf_x") == {"use_auth_token": "hf_x"}

    def test_a_kwargs_only_signature_gets_the_current_spelling(self):
        class PipelineOpaque:
            @staticmethod
            def from_pretrained(checkpoint, **kwargs):
                ...

        assert _token_kwargs(PipelineOpaque, "hf_x") == {"token": "hf_x"}

    def test_an_uninspectable_callable_gets_the_current_spelling(self):
        class PipelineNative:
            from_pretrained = print  # builtin: signature may be unavailable

        assert _token_kwargs(PipelineNative, "hf_x") == {"token": "hf_x"}


class TestPyannoteBackend:
    def test_it_is_a_diarization_backend(self):
        assert issubclass(DiarizationPyannote, Diarization)

    def test_contradictory_speaker_bounds_are_rejected(self):
        backend = DiarizationPyannote()

        with pytest.raises(ValueError, match="min_speakers"):
            import asyncio
            asyncio.run(backend.diarize("audio.wav", min_speakers=5, max_speakers=2))

    def test_a_missing_token_is_reported_before_any_download(self, monkeypatch):
        for variable in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            monkeypatch.delenv(variable, raising=False)

        pytest.importorskip("pyannote.audio")
        backend = DiarizationPyannote()

        with pytest.raises(RuntimeError, match="gated"):
            backend._read_token()

    def test_the_token_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_secret")

        assert DiarizationPyannote()._read_token() == "hf_secret"

    def test_a_custom_token_variable_is_honoured(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("MY_TOKEN", "hf_other")

        assert DiarizationPyannote(token_variable="MY_TOKEN")._read_token() == "hf_other"

    def test_concurrency_must_be_positive(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            DiarizationPyannote(max_concurrency=0)
