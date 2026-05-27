from __future__ import annotations
import itertools
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union

from ragu.chunker.types import Chunk
from ragu.common.logger import logger
from ragu.common.prompts.messages import ChatMessages, render
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.graph.types import Entity, Relation
from ragu.models.llm import LLM
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor


@dataclass
class ChunkContext:
    """
    Tracks extraction state for a single chunk through all pipeline stages.
    """
    chunk: Chunk
    raw_entities: List[str] = field(default_factory=list[str])  # for pyright
    normalized_entities: List[str] = field(default_factory=list[str])
    entities: List[Entity] = field(default_factory=list[Entity])
    relations: List[Relation] = field(default_factory=list[Relation])


class RaguLmArtifactExtractor(BaseArtifactExtractor):
    """
    RAGU-LM artifact extractor with stage-by-stage batch processing.
    """

    def __init__(
        self,
        llm: LLM,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> None:
        """
        Artifact extractor powered by RAGU-LM with optimized batch processing.

        :param client: Language model client.
        :param model_name: Model name used for inference.
        :param temperature: Sampling temperature used in generation.
        :param top_p: Probability mass for nucleus sampling.
        """
        super().__init__(prompts=[
            "ragu_lm_entity_extraction",
            "ragu_lm_entity_normalization",
            "ragu_lm_entity_description",
            "ragu_lm_relation_description",
        ])

        self.llm = llm

        self.temperature = temperature
        self.top_p = top_p

    async def extract(
        self,
        chunks: List[Chunk],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Run optimized knowledge extraction pipeline via RAGU-LM.

        Uses stage-by-stage batch processing for better vLLM utilization:
        1. Extract entities from chunks
        2. Normalize entities across chunks
        3. Generate descriptions for entities
        4. Extract relations for inner product of entities from every chunk

        :param chunks: Text chunks to process.
        :return: Tuple of (entities, relations) extracted from all chunks.
        """
        if not chunks:
            return [], []

        start_time = time.time()

        contexts = [ChunkContext(chunk=chunk) for chunk in chunks]

        try:
            # Stage 1: Extract raw entities from chunks
            logger.info(f"Stage 1/4: Extracting entities from {len(chunks)} chunks...")
            await self._batch_extract_entities(contexts)
            total_raw = sum(len(ctx.raw_entities) for ctx in contexts)
            logger.info(f"Extracted {total_raw} raw entities from {len(chunks)} chunks")

            # Stage 2: Normalize entities
            logger.info(f"Stage 2/4: Normalizing {total_raw} entities...")
            await self._batch_normalize_entities(contexts)
            total_normalized = sum(len(ctx.normalized_entities) for ctx in contexts)
            logger.info(f"Normalized to {total_normalized} entities")

            # Stage 3: Generate descriptions for entities
            logger.info(f"Stage 3/4: Generating descriptions for {total_normalized} entities...")
            await self._batch_generate_descriptions(contexts)
            total_entities = sum(len(ctx.entities) for ctx in contexts)
            logger.info(f"Created {total_entities} entity objects")

            # Stage 4: Extract relations for entity pairs
            total_pairs = sum(len(ctx.entities) * (len(ctx.entities) - 1) for ctx in contexts)
            logger.info(f"Stage 4/4: Extracting relations for {total_pairs} entity pairs...")
            await self._batch_extract_relations(contexts)
            total_relations = sum(len(ctx.relations) for ctx in contexts)
            logger.info(f"Extracted {total_relations} relations")
        except Exception as e:
            logger.warning(
                "RAGU-LM extraction pipeline failed: {}: {}. Returning partial results.",
                type(e).__name__, e,
            )

        all_entities = [e for ctx in contexts for e in ctx.entities]
        all_relations = [r for ctx in contexts for r in ctx.relations]

        elapsed = time.time() - start_time
        logger.info(
            f"Extraction complete: {len(all_entities)} entities, {len(all_relations)} relations "
            f"from {len(chunks)} chunks in {elapsed:.2f}s"
        )

        return all_entities, all_relations

    async def _batch_extract_entities(self, contexts: List[ChunkContext]) -> None:
        """
        Stage 1: Extract raw entities from all chunks in a single batch.
        """
        instruction: RAGUInstruction = self.get_prompt("ragu_lm_entity_extraction")

        conversations: List[ChatMessages] = render(
            instruction.messages,
            text=[ctx.chunk.content for ctx in contexts]
        )

        if not conversations:
            return

        responses = await self._run(conversations, description="Extracting entities.")

        # Parse responses back to contexts
        for ctx, response in zip(contexts, responses):
            if response is None:
                ctx.raw_entities = []
                continue
            lines = response.splitlines()
            entities = [ln.strip() for ln in lines if ln.strip()]
            unique_entities = list(dict.fromkeys(entities))  # Preserve order, remove duplicates

            if len(unique_entities) != len(entities):
                logger.debug(f"Removed {len(entities) - len(unique_entities)} duplicate entities from chunk {ctx.chunk.id}")

            ctx.raw_entities = unique_entities

    async def _batch_normalize_entities(self, contexts: List[ChunkContext]) -> None:
        """
        Stage 2: Normalize all entities across all chunks in a single batch.
        """
        instruction: RAGUInstruction = self.get_prompt("ragu_lm_entity_normalization")

        conversations: list[ChatMessages] = []
        prompt_map: List[Tuple[ChunkContext, int]] = []

        for ctx in contexts:
            if not ctx.raw_entities:
                continue

            chunk_conversations: List[ChatMessages] = render(
                instruction.messages,
                source_text=ctx.chunk.content,
                source_entity=ctx.raw_entities,
            )

            for i, conversation in enumerate(chunk_conversations):
                conversations.append(conversation)
                prompt_map.append((ctx, i))

        if not conversations:
            return

        responses = await self._run(conversations, description="Normalizing entities")

        # Parse responses back to contexts
        for (ctx, _entity_idx), response in zip(prompt_map, responses):
            if response:
                ctx.normalized_entities.append(response)

    async def _batch_generate_descriptions(self, contexts: List[ChunkContext]) -> None:
        """
        Stage 3: Generate descriptions for all entities in a single batch.
        """
        instruction: RAGUInstruction = self.get_prompt("ragu_lm_entity_description")

        conversations: list[ChatMessages] = []
        prompt_map: List[Tuple[ChunkContext, str]] = []  # (context, entity_name)

        for ctx in contexts:
            if not ctx.normalized_entities:
                continue

            chunk_conversations: List[ChatMessages] = render(
                instruction.messages,
                normalized_entity=ctx.normalized_entities,
                source_text=ctx.chunk.content,
            )

            for conversation, entity_name in zip(chunk_conversations, ctx.normalized_entities):
                conversations.append(conversation)
                prompt_map.append((ctx, entity_name))

        if not conversations:
            return

        responses = await self._run(conversations, description="Generating descriptions")

        # Parse responses back to contexts
        for (ctx, entity_name), response in zip(prompt_map, responses):
            if not response:
                continue

            entity = Entity(
                entity_name=entity_name,
                entity_type="UNKNOWN",
                description=response,
                source_chunk_id=[ctx.chunk.id],
                documents_id=[ctx.chunk.doc_id] if getattr(ctx.chunk, "doc_id", None) else [],
                clusters=[],
            )
            ctx.entities.append(entity)

    async def _batch_extract_relations(self, contexts: List[ChunkContext]) -> None:
        """
        Stage 4: Extract relations for all entity pairs in a single batch.
        """
        instruction: RAGUInstruction = self.get_prompt("ragu_lm_relation_description")

        conversations: list[ChatMessages] = []
        prompt_map: List[Tuple[ChunkContext, Entity, Entity]] = []  # (context, subject, object)

        for ctx in contexts:
            if len(ctx.entities) < 2:
                continue

            entity_pairs = list(itertools.permutations(ctx.entities, 2))
            first_entities = [pair[0].entity_name for pair in entity_pairs]
            second_entities = [pair[1].entity_name for pair in entity_pairs]

            chunk_conversations: List[ChatMessages] = render(
                instruction.messages,
                first_normalized_entity=first_entities,
                second_normalized_entity=second_entities,
                source_text=ctx.chunk.content,
            )

            for conversation, (subject, obj) in zip(chunk_conversations, entity_pairs):
                conversations.append(conversation)
                prompt_map.append((ctx, subject, obj))

        if not conversations:
            return

        responses = await self._run(conversations, description="Extracting relations")

        # Parse responses and collect candidates per context
        context_candidates: Dict[int, List[Relation]] = {id(ctx): [] for ctx in contexts}

        for (ctx, subject, obj), response in zip(prompt_map, responses):
            if response is None:
                continue
            assert subject.id and obj.id, (
                'On error here, decide what to do if .id is None,'
                'but .subject_id and .object_id cannot be None'
            )
            relation = Relation(
                subject_id=subject.id,
                object_id=obj.id,
                subject_name=subject.entity_name,
                object_name=obj.entity_name,
                relation_type="UNKNOWN",
                description=response,
                source_chunk_id=[ctx.chunk.id],
            )
            context_candidates[id(ctx)].append(relation)

        # Filter relations per context
        for ctx in contexts:
            candidates = context_candidates[id(ctx)]
            ctx.relations = self.filter_relations(candidates)

    async def _run(self, conversations: List[ChatMessages], description: str = "") -> List[str | None]:
        """
        Run LLM inference on a batch of conversations.

        :param conversations: List of ChatMessages to process.
        :param description: Description for progress bar.
        :return: List of response strings.  ``None`` marks failed calls.
        """
        return self.llm.batch_chat_completion(
            [c.to_openai() for c in conversations],
            output_schema=str,
            continue_on_error=True,
            temperature=self.temperature,
            top_p=self.top_p,
            desc=description,
        )

    @staticmethod
    def filter_relations(
            relations: List[Relation],
            negative_pattern: Optional[Union[str, re.Pattern[str]]] = None,
    ) -> List[Relation]:
        """
        Filter out empty, irrelevant, or negated relations.
        """
        def _clean_bullet(s: str) -> str:
            return re.sub(r"^[\-\u2022]\s*", "", (s or "").strip())

        NEGATION_PATTERNS = [
            r"^\s*$",
            r"^\s*[\-–—]\s*$",
            r"^(?:[-•]\s*)?(?:отсутств\w*\s+(?:связ\w*|отнош\w*)|нет\s+(?:связ\w*|отнош\w*|информац\w*|данн\w*|сведен\w*))\b",
            r"\bтекст\s+не\s+содерж\w*\b",
            r"\b(?:текст\s+)?не\s+содерж\w*\s+информац\w*\s+о\b",
            r"\bнет\s+(?:информац\w*|сведен\w*|данн\w*)(?:\s+о\b|\b)",
            r"\bне\s+явля\w*\s+\w*отнош\w*",
            r"\bнет\s+\w*отнош\w*",
            r"\bотсутств\w*\s+\w*отнош\w*",
            r"\bне\s+содерж\w*\s+\w*отнош\w*",
            r"\bнет\s+явн\w*\s+\w*отнош\w*",
            r"\bнет\s+\w*связ\w*",
            r"\bотсутств\w*\s+\w*связ\w*",
            r"\bсвяз\w*\s+не\s+(?:установ\w*|прослежива\w*|подтвержд\w*|обнаруж\w*)",
            r"\bотнош\w*\s+не\s+(?:установ\w*|прослежива\w*|подтвержд\w*|обнаруж\w*)",
            r"\bне[^.\n]{0,60}(?:содерж\w*|ука\w*|упомина\w*|найд\w*|обнаруж\w*|подтвержд\w*|установ\w*|прослеж\w*)[^.\n]{0,80}(?:связ\w*|отнош\w*|информац\w*)",
        ]

        if isinstance(negative_pattern, re.Pattern):
            neg = negative_pattern
        else:
            neg = re.compile(
                negative_pattern or r"(?:" + "|".join(NEGATION_PATTERNS) + r")",
                flags=re.IGNORECASE | re.UNICODE
            )

        kept: List[Relation] = []
        for rel in relations:
            cleaned = _clean_bullet(rel.description)
            if not cleaned:
                continue
            if neg.search(cleaned):
                continue
            rel.description = cleaned
            kept.append(rel)

        return kept