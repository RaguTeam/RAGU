"""Batch behaviour of the ASR, OCR and diarization backends.

Two separate guarantees are covered here:

- the base implementations bound how much runs at once, instead of handing the
  whole list to ``asyncio.gather``;
- ``OCRTransformers`` goes further and puts one batch through a single
  ``generate`` call, so the batch has to survive as a batch.

A third guarantee sits underneath both: one failed item must not cost the
caller everything that did succeed.
"""
import asyncio

import pytest

from ragu.models.asr import ASR, Transcript
from ragu.models.diarization import Diarization, SpeakerTurn
from ragu.models.ocr import OCR, OCRTransformers


class ConcurrencyTracker:
    """Records how many calls were in flight simultaneously."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.order: list[int] = []

    async def enter(self, index: int) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.order.append(index)
        await asyncio.sleep(0)  # let every started coroutine pile up before release

    def leave(self) -> None:
        self.active -= 1


class TrackingOCR(OCR):
    def __init__(self) -> None:
        self.tracker = ConcurrencyTracker()

    async def recognize(self, image, prompt=None, mime="image/png") -> str:
        await self.tracker.enter(int(image))
        try:
            return f"page-{int(image)}"
        finally:
            self.tracker.leave()


class TrackingASR(ASR):
    def __init__(self) -> None:
        self.tracker = ConcurrencyTracker()

    async def transcribe(self, audio, language=None, prompt=None) -> Transcript:
        await self.tracker.enter(int(audio))
        try:
            return Transcript(text=f"clip-{int(audio)}")
        finally:
            self.tracker.leave()


class TrackingDiarization(Diarization):
    def __init__(self) -> None:
        self.tracker = ConcurrencyTracker()
        self.bounds: list[tuple] = []

    async def diarize(self, audio, num_speakers=None, min_speakers=None, max_speakers=None):
        await self.tracker.enter(int(audio))
        self.bounds.append((num_speakers, min_speakers, max_speakers))
        try:
            return [SpeakerTurn(start=0.0, end=1.0, speaker=f"S{int(audio)}")]
        finally:
            self.tracker.leave()


class FailingOCR(OCR):
    """Recognizes every page but the ones it was told to fail."""

    def __init__(self, failing: set[int]) -> None:
        self.failing = failing
        self.seen: list[int] = []

    async def recognize(self, image, prompt=None, mime="image/png") -> str:
        index = int(image)
        self.seen.append(index)
        if index in self.failing:
            raise RuntimeError(f"page {index} is unreadable")
        return f"page-{index}"


class TestContinueOnError:
    """A corpus is not worth losing to one bad file."""

    async def test_a_failed_page_becomes_none_and_the_rest_survive(self):
        ocr = FailingOCR(failing={2})

        results = await ocr.recognize_batch(
            [b"0", b"1", b"2", b"3"], batch_size=2, continue_on_error=True
        )

        assert results == ["page-0", "page-1", None, "page-3"]

    async def test_failures_keep_the_positions_of_their_inputs(self):
        ocr = FailingOCR(failing={0, 3})

        results = await ocr.recognize_batch(
            [b"0", b"1", b"2", b"3"], batch_size=4, continue_on_error=True
        )

        assert results == [None, "page-1", "page-2", None]

    async def test_later_batches_still_run_after_an_earlier_one_failed(self):
        ocr = FailingOCR(failing={0})

        await ocr.recognize_batch(
            [b"0", b"1", b"2", b"3"], batch_size=2, continue_on_error=True
        )

        assert sorted(ocr.seen) == [0, 1, 2, 3]

    async def test_the_default_still_raises(self):
        # Silently turning failures into None would be the wrong default: a
        # caller who does not ask for it should hear about the failure.
        ocr = FailingOCR(failing={1})

        with pytest.raises(RuntimeError):
            await ocr.recognize_batch([b"0", b"1"], batch_size=2)

    async def test_a_failed_recording_becomes_none(self):
        class FailingASR(ASR):
            async def transcribe(self, audio, language=None, prompt=None):
                if int(audio) == 1:
                    raise RuntimeError("unreadable")
                return Transcript(text=f"clip-{int(audio)}")

        results = await FailingASR().transcribe_batch(
            [b"0", b"1", b"2"], batch_size=2, continue_on_error=True
        )

        assert [r.text if r else None for r in results] == ["clip-0", None, "clip-2"]

    async def test_a_failed_diarization_becomes_none(self):
        class FailingDiarization(Diarization):
            async def diarize(self, audio, num_speakers=None, min_speakers=None,
                              max_speakers=None):
                if int(audio) == 0:
                    raise RuntimeError("no pipeline")
                return [SpeakerTurn(start=0.0, end=1.0, speaker="S0")]

        results = await FailingDiarization().diarize_batch(
            [b"0", b"1"], batch_size=2, continue_on_error=True
        )

        assert results[0] is None
        assert results[1] == [SpeakerTurn(start=0.0, end=1.0, speaker="S0")]

    async def test_the_failure_is_reported(self):
        from ragu.common.logger import logger

        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(message), level="WARNING")
        try:
            await FailingOCR(failing={1}).recognize_batch(
                [b"0", b"1"], batch_size=2, continue_on_error=True
            )
        finally:
            logger.remove(sink)

        assert any("unreadable" in warning for warning in warnings)


class TestLocalOCRBatchFailures:
    """`OCRTransformers` overrides `recognize_batch`; the override has to keep
    the base class's contract, including the options it accepts."""

    @staticmethod
    def _ocr(max_width=None, failing=None):
        ocr = OCRTransformers.__new__(OCRTransformers)
        ocr._load_lock = asyncio.Lock()
        ocr._inference_semaphore = asyncio.Semaphore(1)
        ocr._model = object()
        ocr.prompt = "Text Recognition:"
        calls: list[int] = []

        async def run(images, prompt):
            calls.append(len(images))
            if max_width is not None and len(images) > max_width:
                raise RuntimeError(f"out of memory for {len(images)} pages")
            if failing and failing.intersection(images):
                raise RuntimeError("unreadable page")
            return [f"page-{image.decode()}" for image in images]

        ocr._run = run
        return ocr, calls

    async def test_the_override_accepts_the_base_options(self):
        # Adding desc/continue_on_error to OCR.recognize_batch without adding
        # them here made this call a TypeError.
        ocr, _calls = self._ocr()

        results = await ocr.recognize_batch(
            [b"0", b"1"], batch_size=2, desc=None, continue_on_error=True
        )

        assert results == ["page-0", "page-1"]

    async def test_a_batch_too_wide_recovers_every_page(self):
        ocr, calls = self._ocr(max_width=2)

        results = await ocr.recognize_batch(
            [b"0", b"1", b"2", b"3"], batch_size=4, desc=None, continue_on_error=True
        )

        assert results == ["page-0", "page-1", "page-2", "page-3"]
        assert calls[0] == 4 and calls[1:] == [1, 1, 1, 1]

    async def test_an_unreadable_page_costs_only_itself(self):
        ocr, _calls = self._ocr(failing={b"1"})

        results = await ocr.recognize_batch(
            [b"0", b"1", b"2"], batch_size=3, desc=None, continue_on_error=True
        )

        assert results == ["page-0", None, "page-2"]

    async def test_the_default_still_raises(self):
        ocr, _calls = self._ocr(failing={b"1"})

        with pytest.raises(RuntimeError):
            await ocr.recognize_batch([b"0", b"1"], batch_size=2, desc=None)


class TestBoundedBatching:
    async def test_ocr_never_exceeds_batch_size(self):
        ocr = TrackingOCR()

        results = await ocr.recognize_batch([b"0", b"1", b"2", b"3", b"4"], batch_size=2)

        assert results == ["page-0", "page-1", "page-2", "page-3", "page-4"]
        assert ocr.tracker.peak == 2

    async def test_asr_never_exceeds_batch_size(self):
        asr = TrackingASR()

        results = await asr.transcribe_batch([b"0", b"1", b"2", b"3", b"4"], batch_size=2)

        assert [t.text for t in results] == ["clip-0", "clip-1", "clip-2", "clip-3", "clip-4"]
        assert asr.tracker.peak == 2

    async def test_diarization_never_exceeds_batch_size(self):
        diarization = TrackingDiarization()

        results = await diarization.diarize_batch([b"0", b"1", b"2"], batch_size=1)

        assert [turns[0].speaker for turns in results] == ["S0", "S1", "S2"]
        assert diarization.tracker.peak == 1

    async def test_speaker_bounds_reach_every_recording(self):
        diarization = TrackingDiarization()

        await diarization.diarize_batch([b"0", b"1"], num_speakers=3, batch_size=1)

        assert diarization.bounds == [(3, None, None), (3, None, None)]

    async def test_a_batch_larger_than_the_input_is_harmless(self):
        ocr = TrackingOCR()

        results = await ocr.recognize_batch([b"0", b"1"], batch_size=99)

        assert results == ["page-0", "page-1"]
        assert ocr.tracker.peak == 2

    async def test_empty_input(self):
        assert await TrackingOCR().recognize_batch([]) == []
        assert await TrackingASR().transcribe_batch([]) == []
        assert await TrackingDiarization().diarize_batch([]) == []

    @pytest.mark.parametrize("bad", [0, -1])
    async def test_batch_size_must_be_positive(self, bad):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            await TrackingOCR().recognize_batch([b"0"], batch_size=bad)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            await TrackingASR().transcribe_batch([b"0"], batch_size=bad)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            await TrackingDiarization().diarize_batch([b"0"], batch_size=bad)


class StubInputs(dict):
    """Mimics the processor's BatchFeature: `.to(device)` and a `pop`."""

    def to(self, device):
        return self


class StubProcessor:
    """Counts how many chat templates it is asked to encode at a time."""

    def __init__(self) -> None:
        self.batch_shapes: list[int] = []
        self.tokenizer = type("_Tok", (), {"padding_side": "right"})()

    def apply_chat_template(self, conversations, **kwargs):
        import torch

        self.batch_shapes.append(len(conversations))
        # Prompt text is echoed back by decode() so tests can check routing.
        self._texts = [
            conversation[0]["content"][1]["text"] for conversation in conversations
        ]
        return StubInputs(input_ids=torch.zeros((len(conversations), 4), dtype=torch.long))

    def decode(self, sequence, skip_special_tokens=True):
        import torch

        return f"text-{int(torch.as_tensor(sequence)[0])}"


class StubModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate(self, **kwargs):
        import torch

        self.generate_calls += 1
        rows = kwargs["input_ids"].shape[0]
        # Row i decodes to "text-i" via the stub decoder, so order is checkable.
        return torch.stack([
            torch.cat([torch.zeros(4, dtype=torch.long),
                       torch.full((2,), i, dtype=torch.long)])
            for i in range(rows)
        ])


def _png(colour: int) -> bytes:
    """A real 1x1 PNG — OCRTransformers decodes its input with PIL."""
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (1, 1), (colour, colour, colour)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def stubbed_ocr(monkeypatch):
    """OCRTransformers with the two `from_pretrained` calls stubbed out.

    Patching those rather than `_load` itself keeps the real loading body under
    test, including the left-padding it sets up.
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")

    processor, model = StubProcessor(), StubModel()
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained",
        classmethod(lambda cls, *a, **k: processor),
    )
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText, "from_pretrained",
        classmethod(lambda cls, *a, **k: model),
    )

    def build(**kwargs) -> tuple[OCRTransformers, StubProcessor, StubModel]:
        return OCRTransformers(**kwargs), processor, model

    return build


class TestTransformersTrueBatching:
    async def test_one_generate_call_per_batch(self, stubbed_ocr):
        ocr, processor, model = stubbed_ocr()

        await ocr.recognize_batch([_png(i) for i in range(5)], batch_size=2)

        # Five pages, batch of two: three calls, not five.
        assert model.generate_calls == 3
        assert processor.batch_shapes == [2, 2, 1]

    async def test_results_keep_input_order(self, stubbed_ocr):
        ocr, _, _ = stubbed_ocr()

        results = await ocr.recognize_batch([_png(i) for i in range(3)], batch_size=3)

        assert results == ["text-0", "text-1", "text-2"]

    async def test_single_recognize_uses_the_same_path(self, stubbed_ocr):
        ocr, processor, model = stubbed_ocr()

        result = await ocr.recognize(_png(0))

        assert result == "text-0"
        assert processor.batch_shapes == [1]
        assert model.generate_calls == 1

    async def test_model_is_loaded_once_across_batches(self, stubbed_ocr):
        ocr, _, _ = stubbed_ocr()
        loads = 0
        inner = ocr._load

        def counting_load() -> None:
            nonlocal loads
            loads += 1
            inner()

        ocr._load = counting_load
        await ocr.recognize_batch([_png(i) for i in range(3)], batch_size=1)

        assert loads == 1

    async def test_padding_side_is_forced_left_on_load(self, stubbed_ocr):
        ocr, processor, _ = stubbed_ocr()
        assert processor.tokenizer.padding_side == "right"

        await ocr.recognize(_png(0))

        # Generation continues from the right edge, so shorter prompts in a
        # batch must be padded on the left.
        assert processor.tokenizer.padding_side == "left"

    async def test_prompt_override_reaches_every_page(self, stubbed_ocr):
        ocr, processor, _ = stubbed_ocr()

        await ocr.recognize_batch(
            [_png(0), _png(1)], prompt="Table Recognition:", batch_size=2
        )

        assert processor._texts == ["Table Recognition:", "Table Recognition:"]

    async def test_max_concurrency_must_be_positive(self):
        with pytest.raises(ValueError, match="max_concurrency must be positive"):
            OCRTransformers(max_concurrency=0)

    async def test_inference_is_serialized_by_the_semaphore(self, stubbed_ocr):
        ocr, _, _ = stubbed_ocr(max_concurrency=1)
        tracker = ConcurrencyTracker()
        original = ocr._recognize_sync

        def tracked(images, prompt):
            tracker.active += 1
            tracker.peak = max(tracker.peak, tracker.active)
            try:
                return original(images, prompt)
            finally:
                tracker.active -= 1

        ocr._recognize_sync = tracked
        await asyncio.gather(*(ocr.recognize(_png(i)) for i in range(4)))

        assert tracker.peak == 1
