import asyncio
import base64
from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Any, List, Sequence

from openai.types.chat import ChatCompletionMessageParam

from tqdm import tqdm

from ragu.common.batch_generator import BatchGenerator
from ragu.common.logger import logger
from ragu.utils.batching import release_gpu_memory, run_batched
from ragu.utils.ragu_utils import experimental

if TYPE_CHECKING:
    from ragu.models.openai import CachedAsyncOpenAI


DEFAULT_BATCH_SIZE = 4

# Prompts recognized by GLM-OCR (see https://huggingface.co/zai-org/GLM-OCR).
TEXT_RECOGNITION_PROMPT = "Text Recognition:"
TABLE_RECOGNITION_PROMPT = "Table Recognition:"
FORMULA_RECOGNITION_PROMPT = "Formula Recognition:"

DEFAULT_OCR_PROMPT = TEXT_RECOGNITION_PROMPT

DEFAULT_GLM_OCR_MODEL = "zai-org/GLM-OCR"
DEFAULT_OCR_MAX_TOKENS = 8192


class OCR(ABC):
    """
    Abstract interface for OCR backends.

    An OCR backend converts a page/image (encoded bytes) into structured
    text (Markdown with tables and LaTeX formulas, depending on the model
    and prompt).
    """

    @abstractmethod
    async def recognize(
        self,
        image: bytes,
        prompt: str | None = None,
        mime: str = "image/png",
    ) -> str:
        """
        Recognize a single image.

        :param image: Encoded image bytes (PNG/JPEG).
        :type image: bytes
        :param prompt: Recognition prompt; ``None`` uses the backend default.
        :type prompt: str | None
        :param mime: MIME type of the image bytes.
        :type mime: str
        :returns: Recognized text/markdown.
        :rtype: str
        """

    async def recognize_batch(
        self,
        images: Sequence[bytes],
        prompt: str | None = None,
        mime: str = "image/png",
        batch_size: int = DEFAULT_BATCH_SIZE,
        desc: str | None = "Recognizing",
        continue_on_error: bool = False,
    ) -> List[str | None]:
        """
        Recognize multiple images, `batch_size` at a time.

        :param images: Encoded image bytes for each page.
        :type images: Sequence[bytes]
        :param prompt: Recognition prompt; ``None`` uses the backend default.
        :type prompt: str | None
        :param mime: MIME type of the image bytes.
        :type mime: str
        :param batch_size: Images processed concurrently per batch.
        :type batch_size: int
        :param desc: Progress bar description; a long document should not look
            hung while its pages are recognized.
        :type desc: str | None
        :param continue_on_error: Keep going past a page that failed, returning
            ``None`` in its place, instead of losing five hundred recognized
            pages to the one that was not.
        :type continue_on_error: bool
        :returns: Recognized text for each image, in input order, ``None`` for
            pages that failed when `continue_on_error` is set.
        :rtype: List[str | None]
        :raises ValueError: If `batch_size` is not positive.
        """
        return await run_batched(
            images,
            lambda image: self.recognize(image, prompt=prompt, mime=mime),
            batch_size=batch_size,
            desc=desc,
            continue_on_error=continue_on_error,
        )


@experimental
class OCROpenAI(OCR):
    """
    OCR over an OpenAI-compatible API, defaulting to GLM-OCR (vLLM, SGLang,
    Ollama or the z.ai API).

    Reuses :class:`~ragu.models.openai.CachedAsyncOpenAI`, so responses are
    cached, rate limited and retried the same way as regular LLM calls.

    Example::

        from ragu.models.openai import CachedAsyncOpenAI
        from ragu.models.ocr import OCROpenAI

        client = CachedAsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        ocr = OCROpenAI(client=client, model_name="zai-org/GLM-OCR")
    """

    def __init__(
        self,
        client: "CachedAsyncOpenAI",
        model_name: str = DEFAULT_GLM_OCR_MODEL,
        prompt: str = DEFAULT_OCR_PROMPT,
        max_tokens: int | None = DEFAULT_OCR_MAX_TOKENS,
        **generation_kwargs: Any,
    ) -> None:
        """
        Initialize the remote OCR backend.

        :param client: OpenAI-compatible client used for chat completions.
        :type client: CachedAsyncOpenAI
        :param model_name: Model name registered on the serving endpoint.
        :type model_name: str
        :param prompt: Default recognition prompt (``"Text Recognition:"``,
            ``"Table Recognition:"`` or ``"Formula Recognition:"``).
        :type prompt: str
        :param max_tokens: Generation cap per page, the counterpart of
            :class:`OCRTransformers`'s ``max_new_tokens``. ``None`` leaves the
            endpoint's own default in place, which on a dense page can cut the
            recognized text off mid-table without saying so.
        :type max_tokens: int | None
        :param generation_kwargs: Extra generation options forwarded to the
            chat completion call (for example ``temperature``).
        """
        self.client = client
        self.model_name = model_name
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.generation_kwargs = generation_kwargs

    async def recognize(
        self,
        image: bytes,
        prompt: str | None = None,
        mime: str = "image/png",
    ) -> str:
        """
        Recognize a single image via the OpenAI-compatible endpoint.

        :param image: Encoded image bytes (PNG/JPEG).
        :type image: bytes
        :param prompt: Recognition prompt; ``None`` uses the default prompt.
        :type prompt: str | None
        :param mime: MIME type of the image bytes.
        :type mime: str
        :returns: Recognized text/markdown.
        :rtype: str
        """
        encoded = base64.b64encode(image).decode("ascii")
        conversation: list[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                    {"type": "text", "text": prompt or self.prompt},
                ],
            }
        ]
        options: dict[str, Any] = dict(self.generation_kwargs)
        if self.max_tokens is not None:
            options.setdefault("max_tokens", self.max_tokens)

        response = await self.client.chat_completion(
            model_name=self.model_name,
            conversation=conversation,
            **options,
        )
        return (response or "").strip()


@experimental
class OCRTransformers(OCR):
    """
    Local OCR backend running through HuggingFace ``transformers``, defaulting
    to GLM-OCR.

    Requires the ``local`` extra (``pip install graph_ragu[local]``).
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_GLM_OCR_MODEL,
        prompt: str = DEFAULT_OCR_PROMPT,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 8192,
        max_concurrency: int = 1,
    ) -> None:
        """
        Initialize the local OCR backend.

        :param model_name_or_path: HF model id or local checkpoint path.
        :type model_name_or_path: str
        :param prompt: Default recognition prompt.
        :type prompt: str
        :param device_map: ``device_map`` passed to ``from_pretrained``.
        :type device_map: str
        :param dtype: ``dtype`` passed to ``from_pretrained``.
        :type dtype: str
        :param max_new_tokens: Generation cap per page.
        :type max_new_tokens: int
        :param max_concurrency: Number of concurrent inference calls allowed.
        :type max_concurrency: int
        :raises ValueError: If `max_concurrency` is not positive.
        """
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")

        self.model_name_or_path = model_name_or_path
        self.prompt = prompt
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = asyncio.Lock()
        self._inference_semaphore = asyncio.Semaphore(max_concurrency)

    async def recognize(
        self,
        image: bytes,
        prompt: str | None = None,
        mime: str = "image/png",
    ) -> str:
        """
        Recognize a single image with the local model.

        :param image: Encoded image bytes (PNG/JPEG).*
        :type image: bytes
        :param prompt: Recognition prompt; ``None`` uses the default prompt.
        :type prompt: str | None
        :param mime: MIME type of the image bytes (unused, the image is
            decoded with PIL).
        :type mime: str
        :returns: Recognized text/markdown.
        :rtype: str
        :raises ImportError: If ``transformers`` or ``pillow`` are missing.
        """
        recognized = await self._run([image], prompt or self.prompt)
        return recognized[0]

    async def recognize_batch(
        self,
        images: Sequence[bytes],
        prompt: str | None = None,
        mime: str = "image/png",
        batch_size: int = DEFAULT_BATCH_SIZE,
        desc: str | None = "Recognizing",
        continue_on_error: bool = False,
    ) -> List[str | None]:
        """
        Recognize multiple images, one ``generate`` call per batch.

        :param images: Encoded image bytes for each page.
        :type images: Sequence[bytes]
        :param prompt: Recognition prompt; ``None`` uses the default prompt.
        :type prompt: str | None
        :param mime: MIME type of the image bytes (unused, images are decoded
            with PIL).
        :type mime: str
        :param batch_size: Pages per ``generate`` call. Raise it for throughput,
            lower it when VRAM is tight.
        :type batch_size: int
        :param desc: Progress bar description.
        :type desc: str | None
        :param continue_on_error: Keep going past a page that failed, returning
            ``None`` in its place. A batch that fails as a whole is retried one
            page at a time, so neither an unreadable page nor a batch too wide
            for the card costs the rest.
        :type continue_on_error: bool
        :returns: Recognized text for each image, in input order, ``None`` for
            pages that failed when `continue_on_error` is set.
        :rtype: List[str | None]
        :raises ValueError: If `batch_size` is not positive.
        :raises ImportError: If ``transformers`` or ``pillow`` are missing.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        pages = list(images)
        effective_prompt = prompt or self.prompt

        results: List[str | None] = []
        with tqdm(total=len(pages), desc=desc) as progress:
            for batch in BatchGenerator(pages, batch_size).get_batches():
                results.extend(await self._recognize_group(
                    list(batch), effective_prompt, continue_on_error, progress,
                ))

        return results

    async def _recognize_group(
        self,
        batch: List[bytes],
        prompt: str,
        continue_on_error: bool,
        progress: Any,
    ) -> List[str | None]:
        """
        Put one batch through a single ``generate`` call.

        :param batch: Encoded image bytes forming the batch.
        :param prompt: Recognition prompt applied to every page.
        :param continue_on_error: Fall back to one page at a time when the batch
            fails as a whole, instead of raising.
        :param progress: Progress bar to advance.
        :returns: Recognized text in batch order.
        :rtype: List[str | None]
        """
        recognized: List[str] | None = None
        failure: str | None = None

        try:
            recognized = await self._run(batch, prompt)
        except Exception as error:  # noqa: BLE001 - the policy decides what this means
            if not continue_on_error:
                raise
            failure = f"{type(error).__name__}: {error}"

        if failure is not None:
            logger.warning(
                "A batch of {} pages failed ({}); retrying them one at a time",
                len(batch), failure,
            )
            await asyncio.to_thread(release_gpu_memory)
            return await self._recognize_singly(batch, prompt, progress)

        progress.update(len(batch))
        return list(recognized or ())

    async def _recognize_singly(
        self,
        batch: List[bytes],
        prompt: str,
        progress: Any,
    ) -> List[str | None]:
        """
        Recognize a failed batch one page at a time.

        :param batch: Encoded image bytes to retry.
        :param prompt: Recognition prompt applied to every page.
        :param progress: Progress bar to advance.
        :returns: Recognized text in batch order, ``None`` where one failed.
        :rtype: List[str | None]
        """
        results: List[str | None] = []

        for index, image in enumerate(batch):
            try:
                results.append((await self._run([image], prompt))[0])
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                logger.warning(
                    "Recognition failed for page {} of the batch: {}: {}",
                    index, type(error).__name__, error,
                )
                results.append(None)
            finally:
                progress.update(1)

        return results

    async def _run(self, images: List[bytes], prompt: str) -> List[str]:
        """
        Load on first use, then run one batch through the model.

        :param images: Encoded image bytes forming a single batch.
        :type images: List[bytes]
        :param prompt: Recognition prompt applied to every image.
        :type prompt: str
        :returns: Recognized text, in input order.
        :rtype: List[str]
        """
        async with self._load_lock:
            if self._model is None:
                await asyncio.to_thread(self._load)
        async with self._inference_semaphore:
            return await asyncio.to_thread(self._recognize_sync, images, prompt)

    def _load(self) -> None:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for OCRTransformers. "
                "Install it with: pip install graph_ragu[local]"
            ) from exc

        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            dtype=self.dtype,
            device_map=self.device_map,
        )

        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    def _recognize_sync(self, images: List[bytes], prompt: str) -> List[str]:
        """
        Run one batch of images through the model, blocking.

        :param images: Encoded image bytes forming a single batch.
        :type images: List[bytes]
        :param prompt: Recognition prompt applied to every image.
        :type prompt: str
        :returns: Recognized text, in input order.
        :rtype: List[str]
        :raises ImportError: If ``pillow`` is missing.
        """
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "pillow is required for OCRTransformers. "
                "Install it with: pip install graph_ragu[local]"
            ) from exc

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.open(BytesIO(image)).convert("RGB")},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for image in images
        ]
        inputs = self._processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)
        inputs.pop("token_type_ids", None)

        generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        prompt_length = inputs["input_ids"].shape[1]
        return [
            self._processor.decode(
                sequence[prompt_length:],
                skip_special_tokens=True,
            ).strip()
            for sequence in generated_ids
        ]
