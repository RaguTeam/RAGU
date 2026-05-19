"""
Pydantic schema definitions for structured outputs of the RAGU prompt system.

These models describe the expected structured responses from the LLM
for various prompt types — such as artifact extraction, community analysis,
entity/relation summarization, and global/local search results.

They serve as type-safe contracts between LLM generations and the downstream
processing pipeline (e.g., graph construction, ranking, reporting).

Modules and models overview
---------------------------

• **EntityModel**, **RelationModel**, **ArtifactsModel**
    Define the structure for extracted entities and relationships
    produced during artifact extraction from text.

• **CommunityFindingModel**, **CommunityReportModel**
    Describe the structure of automatically generated reports summarizing
    clusters or communities discovered in a knowledge graph.

• **GlobalSearchContextModel**, **GlobalSearchResponseModel**
    Define the format of reasoning and response outputs during global
    retrieval-augmented search across multiple graph communities.

• **DefaultResponseModel**
    Provides a minimal schema for generic responses.

• **EntityDescriptionModel**, **RelationDescriptionModel**
    Contain simplified summaries of entities and relations for
    documentation and human-readable output.

Each schema includes field-level descriptions to guide the LLM
in producing well-structured, interpretable results.
"""

import logging
from typing import List, Optional, Literal

from pydantic import (
    BaseModel,
    Field,
    conint,
    confloat,
    field_validator,
    model_validator,
)


class EntityModel(BaseModel):
    entity_name: str = Field(..., description="Normalized entity name, capitalized")
    entity_type: str = Field(..., description="Entity type")
    description: str = Field(..., description="Detailed description of the entity from the text")

    @field_validator("entity_type")
    @classmethod
    def _normalize_entity_type(cls, v: str) -> str:
        return v.split(" (")[0].strip() if v else "UNKNOWN"


class RelationModel(BaseModel):
    source_entity: str = Field(..., description="Name of the source entity (matches an Entity.entity_name)")
    target_entity: str = Field(..., description="Name of the target entity (matches an Entity.entity_name)")
    relation_type: str = Field(..., description="Type of relation")
    description: str = Field(..., description="Description of the relationship")
    relationship_strength: conint(ge=0, le=5) = Field(
        ..., description="Relationship strength 0–5 (0 = weak, 5 = strong)"
    )

    @field_validator("relation_type")
    @classmethod
    def _normalize_relation_type(cls, v: str) -> str:
        return v.split(" (")[0].strip() if v else "UNKNOWN"


class ArtifactsModel(BaseModel):
    entities: List[EntityModel] = Field(
        default_factory=list[EntityModel],
        description="List of extracted entities"
    )
    relations: List[RelationModel] = Field(
        default_factory=list[RelationModel],
        description="List of extracted relationships"
    )

    @model_validator(mode="after")
    def _validate_relationship_endpoints(self):
        names = {e.entity_name for e in self.entities}
        bad = []
        for r in self.relations:
            if r.source_entity not in names or r.target_entity not in names:
                bad.append((r.source_entity, r.target_entity))
        if bad:
            logging.warning(
                "Relations reference entities not present in 'entities': "
                + "; ".join(f"{s} -> {t}" for s, t in bad)
            )
        return self


class EntitiesExtractionModel(BaseModel):
    entities: List[EntityModel] = Field(
        default_factory=list,
        description="List of entities extracted from text"
    )


class RelationsExtractionModel(BaseModel):
    relations: List[RelationModel] = Field(
        default_factory=list,
        description="List of relationships between provided entities"
    )


class CommunityFindingModel(BaseModel):
    summary: str = Field(..., description="Short description of the finding")
    explanation: str = Field(..., description="Detailed explanation (several paragraphs based on the data)")


class CommunityReportModel(BaseModel):
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Short summary of the community")
    rating: confloat(ge=0, le=10) = Field(..., description="Impact rating 0–10")
    rating_explanation: str = Field(..., description="Explanation of the rating")
    findings: List[CommunityFindingModel] = Field(
        default_factory=list,
        description="List of 5–10 key findings"
    )


class GlobalSearchContextModel(BaseModel):
    reasoning: str = Field(..., description="Reasoning about the relevance of the context")
    response: str = Field(..., description="Answer to the query")
    rating: confloat(ge=0, le=10) = Field(..., description="Relevance rating of the context 0–10")


class EntityDescriptionModel(BaseModel):
    entity_name: str = Field(description="Entity name")
    description: str = Field(description="Summarized description")


class RelationDescriptionModel(BaseModel):
    subject_name: str = Field(description="Subject entity name")
    object_name: str = Field(description="Object entity name")
    description: str = Field(description="Summarized description of the relationship between the entities")


class ClusterSummarizationModel(BaseModel):
    content: str = Field(description="Summarized content of the cluster")

class SubQuery(BaseModel):
    id: str = Field(..., description="Unique identifier of the subquery, e.g. 'q1', 'q2'")
    query: str = Field(..., description="Natural language formulation of the atomic subquery")
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of subquery IDs that must be resolved before this one"
    )
    intent: Optional[Literal[
        "lookup",
        "definition",
        "comparison",
        "aggregation",
        "reasoning",
        "filtering",
        "other"
    ]] = Field(
        default=None,
        description="Optional classification of the subquery intent"
    )

class QueryPlan(BaseModel):
    subqueries: List[SubQuery] = Field(..., description="List of decomposed subqueries forming a DAG")

class RewriteQuery(BaseModel):
    query: str = Field(..., description="Rewritten query that is self-contained and explicit")
