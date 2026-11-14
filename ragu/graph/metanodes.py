"""Document objects represented as ordinary graph entities."""

from collections import defaultdict
from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessageParam

from ragu.chunker.types import Chunk, FormulaChunk, ObjectChunk, TableChunk
from ragu.common.base import RaguGenerativeModule
from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.common.prompts.default_models import RelationsExtractionModel
from ragu.common.prompts.messages import render
from ragu.common.prompts.prompt_storage import require_prompt_schema
from ragu.graph.types import Entity, Relation
from ragu.models.llm import LLM
from ragu.utils.ragu_utils import compute_mdhash_id
from ragu.utils.token_truncation import TokenTruncation


@dataclass(slots=True)
class _RelationRequest:
    """
    Keep a rendered request together with its relation validation context.

    :param conversation: Messages sent to the model.
    :param metanode: Object represented by this request.
    :param candidates: Allowed text entities keyed by ID.
    :param context_ids: Source IDs of the surrounding text chunks.
    """

    conversation: list[ChatCompletionMessageParam]
    metanode: Entity
    candidates: dict[str, Entity]
    context_ids: set[str]


class NearbyRelationExtractor(RaguGenerativeModule):
    """
    Represent tables and formulas as entities and link them to surrounding text.

    :param llm: Optional model for links; objects are retained without a model.
    :param context_window: Number of text chunks on each side, within the section.
    :raises ValueError: If the context window is not a positive integer.
    """

    def __init__(self, llm: LLM | None, context_window: int = 1) -> None:
        """
        Initialize the object relation extractor.

        :param llm: Optional model for extracting object-to-text relations.
        :param context_window: Maximum text chunks on each side of an object.
        :raises ValueError: If the context window is not a positive integer.
        """
        super().__init__(prompts=["table_relations", "formula_relations"])
        if isinstance(context_window, bool) or not isinstance(context_window, int) or context_window < 1:
            raise ValueError("context_window must be a positive integer")
        self.llm = llm
        self.context_window = context_window

    async def a_extract_nearby(
        self,
        chunks: list[Chunk],
        entities: list[Entity],
        relations: list[Relation],
    ) -> tuple[list[Entity], list[Relation]]:
        """
        Build object entities and extract links to the existing text subgraph.

        :param chunks: Ordered source chunks, including tables and formulas.
        :param entities: Merged entities extracted from text chunks.
        :param relations: Merged relations extracted from text chunks.
        :return: Object entities and validated object-to-text relations.
        """
        objects: list[tuple[Chunk, Entity]] = []
        for chunk in chunks:
            if not isinstance(chunk, (TableChunk, FormulaChunk)):
                continue
            objects.append((chunk, self._create_metanode(chunk)))
        metanodes = [entity for _, entity in objects]
        if not objects or self.llm is None:
            return metanodes, []

        chunks_by_document: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_document[chunk.doc_id].append(chunk)
        for document in chunks_by_document.values():
            document.sort(key=lambda chunk: chunk.chunk_order_idx)
        entities_by_chunk: dict[str, list[Entity]] = defaultdict(list)
        for entity in entities:
            for chunk_id in entity.source_chunk_id:
                entities_by_chunk[chunk_id].append(entity)

        truncation: TokenTruncation | None = None
        requests_by_prompt: dict[str, list[_RelationRequest]] = defaultdict(list)
        for chunk, metanode in objects:
            nearby = self._find_nearby_chunks(chunk, chunks_by_document[chunk.doc_id])
            candidates = {
                entity.id: entity
                for item in nearby
                for entity in entities_by_chunk[item.id]
            }
            if not candidates:
                continue
            prompt_name = "table_relations" if isinstance(chunk, TableChunk) else "formula_relations"
            conversation = self._render_conversation(
                prompt_name, chunk, metanode, nearby, candidates, relations
            )
            # Skip oversized requests rather than silently dropping supplied context.
            prompt_text = "\n".join(str(message.get("content", "")) for message in conversation)
            if truncation is None:
                truncation = TokenTruncation(
                    model_id=Settings.tokenizer_llm_name,
                    tokenizer_type=Settings.tokenizer_llm_backend,
                    max_tokens=Settings.llm_context_token_limit,
                )
            if truncation(prompt_text) != prompt_text:
                logger.warning("Skipping oversized relation context for object {}", metanode.id)
                continue
            requests_by_prompt[prompt_name].append(_RelationRequest(
                conversation=conversation,
                metanode=metanode,
                candidates=candidates,
                context_ids={item.id for item in nearby},
            ))

        edges: dict[str, Relation] = {}
        for prompt_name, requests in requests_by_prompt.items():
            results = await self.llm.batch_chat_completion(
                [request.conversation for request in requests],
                output_schema=RelationsExtractionModel,
                continue_on_error=True,
                desc="Table-to-text relations" if prompt_name == "table_relations" else "Formula-to-text relations",
            )
            for request, result in zip(requests, results, strict=True):
                if result is None:
                    continue
                for edge in self._validate_relations(request, result):
                    if edge.id in edges:
                        previous = edges[edge.id]
                        edge.source_chunk_id = sorted(
                            set(edge.source_chunk_id) | set(previous.source_chunk_id)
                        )
                    edges[edge.id] = edge
        return metanodes, list(edges.values())

    def _find_nearby_chunks(self, chunk: Chunk, document: list[Chunk]) -> list[Chunk]:
        """
        Select surrounding text within the object's section.

        :param chunk: Object whose neighbors are selected.
        :param document: Source document chunks sorted by their position.
        :return: Nearby text chunks in source order.
        """
        index = next(i for i, item in enumerate(document) if item.id == chunk.id)
        nearby: list[Chunk] = []
        for direction in (-1, 1):
            count = 0
            position = index + direction
            while 0 <= position < len(document) and count < self.context_window:
                item = document[position]
                if item.metadata.get("section") != chunk.metadata.get("section"):
                    break
                if not isinstance(item, ObjectChunk):
                    nearby.append(item)
                    count += 1
                position += direction
        nearby.sort(key=lambda item: item.chunk_order_idx)
        return nearby

    def _render_conversation(
        self,
        prompt_name: str,
        chunk: Chunk,
        metanode: Entity,
        nearby: list[Chunk],
        candidates: dict[str, Entity],
        relations: list[Relation],
    ) -> list[ChatCompletionMessageParam]:
        """
        Render the object and its text subgraph into a linking request.

        :param prompt_name: Registered table or formula relation prompt.
        :param chunk: Source object chunk.
        :param metanode: Graph entity representing the object.
        :param nearby: Surrounding text chunks in source order.
        :param candidates: Allowed text entities keyed by ID.
        :param relations: Existing relations from text extraction.
        :return: Messages in the OpenAI chat format.
        :raises ValueError: If the prompt has an incompatible output schema.
        """
        instruction = self.get_prompt(prompt_name)
        require_prompt_schema(instruction, prompt_name, RelationsExtractionModel)
        payload = {
            "object": {
                "id": metanode.id,
                "name": metanode.entity_name,
                "type": metanode.entity_type,
                "description": metanode.description,
                "caption": chunk.metadata.get("caption", ""),
                "expression": chunk.content if isinstance(chunk, FormulaChunk) else "",
                "chunk_order_idx": chunk.chunk_order_idx,
            },
            "before_chunks": [
                {"id": item.id, "order": item.chunk_order_idx, "text": item.to_llm()}
                for item in nearby if item.chunk_order_idx < chunk.chunk_order_idx
            ],
            "after_chunks": [
                {"id": item.id, "order": item.chunk_order_idx, "text": item.to_llm()}
                for item in nearby if item.chunk_order_idx > chunk.chunk_order_idx
            ],
            "entities": [
                {
                    "id": entity.id,
                    "name": entity.entity_name,
                    "type": entity.entity_type,
                    "description": entity.description,
                }
                for entity in candidates.values()
            ],
            "relations": [
                relation.to_dict()
                for relation in relations
                if relation.subject_id in candidates and relation.object_id in candidates
            ],
        }
        return render(instruction.messages, context=payload)[0].to_openai()

    @staticmethod
    def _validate_relations(
        request: _RelationRequest,
        result: RelationsExtractionModel,
    ) -> list[Relation]:
        """
        Resolve endpoint names and keep valid object-to-text relations.

        :param request: Object, candidate entities and source context sent to the model.
        :param result: Relations proposed by the model.
        :return: Validated relations with source chunk references.
        """
        metanode = request.metanode
        candidates = request.candidates
        validated: list[Relation] = []
        endpoints_by_name: dict[str, list[Entity]] = defaultdict(list)
        for entity in [metanode, *candidates.values()]:
            endpoints_by_name[entity.entity_name].append(entity)
        for proposal in result.relations:
            sources = endpoints_by_name.get(proposal.source_entity, [])
            targets = endpoints_by_name.get(proposal.target_entity, [])
            if len(sources) != 1 or len(targets) != 1:
                logger.warning("Rejected relation with an unknown or ambiguous name for object {}", metanode.id)
                continue
            source, target = sources[0], targets[0]
            other_id = target.id if source.id == metanode.id else source.id
            if (
                metanode.id not in (source.id, target.id)
                or other_id not in candidates
                or not proposal.relation_type.strip()
                or proposal.relation_type.upper() == "UNKNOWN"
                or not proposal.description.strip()
            ):
                logger.warning("Rejected invalid relation for object {}", metanode.id)
                continue
            edge = Relation(
                subject_id=source.id,
                object_id=target.id,
                subject_name=source.entity_name,
                object_name=target.entity_name,
                relation_type="_".join(proposal.relation_type.upper().replace("-", " ").split()),
                description=proposal.description.strip(),
                relation_strength=proposal.relationship_strength,
                source_chunk_id=sorted(request.context_ids | set(metanode.source_chunk_id)),
            )
            validated.append(edge)
        return validated

    @staticmethod
    def _create_metanode(chunk: TableChunk | FormulaChunk) -> Entity:
        """
        Create a deterministic entity representing a table or formula.

        :param chunk: Source table or formula chunk.
        :return: Metanode referencing the source chunk and document.
        """
        kind = "TABLE" if isinstance(chunk, TableChunk) else "FORMULA"
        return Entity(
            id=compute_mdhash_id(chunk.id, prefix="meta-"),
            entity_name=chunk.metadata.get("caption") or f"{kind} {chunk.chunk_order_idx + 1}",
            entity_type=kind,
            description=chunk.to_embed(),
            source_chunk_id=[chunk.id],
            documents_id=[chunk.doc_id],
        )
