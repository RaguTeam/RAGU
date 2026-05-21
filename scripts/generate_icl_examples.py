"""
Generate in-context learning examples for RAGU extractors.

This script synthesizes or loads a corpus of texts, generates artifacts
using RAGU's built-in extractors, evaluates quality using an LLM judge
with structured output, and saves examples to JSON files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from lingua import Language, LanguageDetectorBuilder
import yaml


sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field, conint

from ragu.common.logger import logger
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet.llm_artifact_extractor import ArtifactsExtractorLLM
from ragu.triplet.two_stage_extractor import TwoStageArtifactsExtractorLLM


class JudgeRatingModel(BaseModel):
    rating: conint(ge=1, le=10) = Field(
        ...,
        description="Overall quality rating from 1 (poor) to 10 (excellent)",
    )
    explanation: str = Field(
        ...,
        description="Brief 1-2 sentence explanation of the rating",
    )


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    :param config_path: Path to YAML configuration file.
    :return: Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def _expand_value(value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith("${") and "}" in value:
                var_name = value[2:-1]
                return os.getenv(var_name, value)
            elif value.startswith("$") and not value.startswith("${"):
                var_name = value[1:]
                return os.getenv(var_name, value)
            return value
        elif isinstance(value, dict):
            return {k: _expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_expand_value(v) for v in value]
        return value

    return _expand_value(config)


def _build_synthesis_prompt(
    domain: str,
    difficulty: str,
    language: str,
) -> str:
    """
    Build prompt for text synthesis.

    :param domain: Domain for text synthesis.
    :param difficulty: Difficulty level.
    :param language: Target language.
    :return: Synthesis prompt.
    """
    if language == "russian":
        prompt = f"""
Сгенерируйте текст на русском языке для области знаний "{domain}" с уровнем сложности "{difficulty}".

Требования:
- Длина текста: 150-300 слов
- Текст должен содержать 3-8 упоминаний именованных сущностей (люди, организации, места, события, продукты, технологии и т.д.) и 2-6 связей между ними
- Сущности должны быть разнообразными — люди, компании, географические объекты, события, продукты и т.д.
- Текст должен быть реалистичным, осмысленным и связным
- Текст должен быть похож на обычную энциклопедическую или новостную статью
- Избегайте слишком общих или неопределённых сущностей
- НЕ добавляйте никакие аннотации, пометки типов, комментарии или пояснения в скобках рядом с сущностями

Верните только чистый текст без дополнительных комментариев.
"""
    else:
        prompt = f"""
Generate a text in English for domain "{domain}" with difficulty level "{difficulty}".

Requirements:
- Text length: 150-300 words
- The text should naturally mention 3-8 named entities (people, organizations, places, events, products, technologies, etc.) with 2-6 connections between them
- Entities should be diverse — people, companies, geographic locations, events, products, etc.
- Text should be realistic, meaningful, and coherent
- Text should read like a normal encyclopedic or news article
- Avoid overly generic or ambiguous entities
- Do NOT add any type annotations, labels, brackets, or comments next to entities in the text

Return only the plain text without any additional comments.
"""
    return prompt.strip()


def load_input_texts(path: str, languages: list[str]) -> list[dict]:
    """
    Load texts from a file or directory for example generation.

    :param path: Path to a .txt file, .json file, or directory of .txt files.
    :return: List of dicts with text, domain, difficulty, language keys.
    """

    prepared_languages = [Language(it.upper()) for it in languages]
    detector = LanguageDetectorBuilder.from_languages(*prepared_languages).with_preloaded_language_models().build()

    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"Input path does not exist: {path}")

    results: list[dict] = []

    if path_obj.is_dir():
        for txt_file in sorted(path_obj.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8").strip()
            if text:
                results.append({
                    "text": text,
                    "domain": txt_file.stem,
                    "difficulty": "medium",
                    "language": detector.detect_language_of(text).name.lower(),
                })
    elif path_obj.suffix == ".json":
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("texts", [])
        for item in items:
            if isinstance(item, str):
                results.append({
                    "text": item,
                    "domain": "general",
                    "difficulty": "medium",
                    "language": detector.detect_language_of(item).name.lower(),
                })
            elif isinstance(item, dict):
                results.append({
                    "text": item["text"],
                    "domain": item.get("domain", "general"),
                    "difficulty": item.get("difficulty", "medium"),
                    "language": item.get("language", detector.detect_language_of(item["text"]).name.lower()),
                })
    elif path_obj.suffix == ".txt":
        text = path_obj.read_text(encoding="utf-8").strip()
        if text:
            results.append({
                "text": text,
                "domain": path_obj.stem,
                "difficulty": "medium",
                "language": detector.detect_language_of(text).name.lower(),
            })
    else:
        raise ValueError(f"Unsupported input format: {path_obj.suffix}")

    logger.info(f"Loaded {len(results)} texts from {path}")
    return results


def deduplicate_corpus(corpus: list[dict]) -> list[dict]:
    """
    Remove duplicate texts from corpus based on exact content match.

    :param corpus: List of corpus items.
    :return: Deduplicated list preserving insertion order.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for item in corpus:
        text = item["text"].strip()
        if text not in seen:
            seen.add(text)
            unique.append(item)
    duplicates = len(corpus) - len(unique)
    if duplicates:
        logger.info(f"Removed {duplicates} duplicate texts from corpus")
    return unique


def load_existing_texts(output_path: str, prompt_type: str) -> set[str]:
    """
    Load input texts already saved in examples file for a given prompt type.

    :param output_path: Directory containing example JSON files.
    :param prompt_type: Prompt type to load examples for.
    :return: Set of stripped input texts from existing examples.
    """
    filepath = os.path.join(output_path, f"{prompt_type}_examples.json")
    if not os.path.exists(filepath):
        return set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            ex["input_text"].strip()
            for ex in data.get("examples", [])
        }
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Could not read existing examples from {filepath}: {e}")
        return set()


def _text_preview(text: str, max_len: int = 120) -> str:
    """Return a truncated single-line preview of text for logging."""
    single = text.replace("\n", " ")
    if len(single) > max_len:
        return single[:max_len] + "..."
    return single


def _build_judge_prompt(example: dict, prompt_type: str) -> str:
    """
    Build prompt for quality evaluation.

    :param example: Example dictionary.
    :param prompt_type: Prompt type to tailor evaluation criteria.
    :return: Judge prompt.
    """
    output = example["output"]

    entity_count = len(output.get("entities", []))
    relation_count = len(output.get("relations", []))

    entities_list = "\n".join([
        f"   - {e.get('entity_name', 'Unknown')} ({e.get('entity_type', 'Unknown')}): "
        f"{e.get('description', 'No description')[:80]}"
        for e in output.get("entities", [])
    ])

    relations_list = "\n".join([
        f"   - {r.get('source_entity', 'Unknown')} → {r.get('target_entity', 'Unknown')} "
        f"({r.get('relation_type', 'Unknown')}): {r.get('description', 'No description')[:80]}"
        for r in output.get("relations", [])
    ])

    has_entities = prompt_type in (
        "artifact_extraction", "artifact_validation",
        "entity_extraction", "entity_validation",
        "relation_extraction", "relation_validation",
    )
    has_relations = prompt_type in (
        "artifact_extraction", "artifact_validation",
        "relation_extraction", "relation_validation",
    )

    task_description = "entity and relation extraction"
    if has_entities and not has_relations:
        task_description = "entity extraction"
    elif has_relations and not has_entities:
        task_description = "relation extraction"

    criteria_lines = [
        "1. Accuracy: Are all extracted items correct and grounded in the text? "
        "Check each claim against the source text. Penalize hallucinations.",
        "2. Completeness: Did it miss obvious items that should have been extracted?",
        "3. Quality: Are types, descriptions, and other fields appropriate and informative?",
    ]
    if has_entities:
        criteria_lines.append(
            f"4. Entities: Are entity names properly normalized? "
            f"Are descriptions detailed and self-contained? ({entity_count} extracted)"
        )
    if has_relations:
        criteria_lines.append(
            f"5. Relations: Do relation endpoints match actual entity names? "
            f"Are relationship types appropriate? ({relation_count} extracted)"
        )

    criteria = "\n".join(criteria_lines)

    data_lines = [f"**Input Text:**\n{example['input_text']}\n"]
    if has_entities:
        data_lines.append(
            f"\n**Extracted Entities ({entity_count}):**\n{entities_list}"
        )
    if has_relations:
        data_lines.append(
            f"\n**Extracted Relations ({relation_count}):**\n{relations_list}"
        )
    data_section = "\n".join(data_lines)

    prompt = f"""
Evaluate the quality of this {task_description} example.

{data_section}

**Quality Criteria:**
{criteria}

**Rating Scale:**
- 1-3: Poor quality (many errors, incomplete, hallucinations)
- 4-6: Acceptable quality (some errors, mostly correct, minor issues)
- 7-8: Good quality (minor issues, mostly accurate, well-grounded)
- 9-10: Excellent quality (no errors, highly accurate, perfect extraction)
"""
    return prompt.strip()


def _has_entities(prompt_type: str) -> bool:
    return prompt_type in (
        "artifact_extraction", "artifact_validation",
        "entity_extraction", "entity_validation",
        "relation_extraction", "relation_validation",
    )


def _has_relations(prompt_type: str) -> bool:
    return prompt_type in (
        "artifact_extraction", "artifact_validation",
        "relation_extraction", "relation_validation",
    )


def _clean_and_validate_output(
    output: dict,
    prompt_type: str,
) -> tuple[bool, dict]:
    """
    Filter relations with endpoints not in entity list and check non-empty.

    :param output: Output dict with entities and/or relations.
    :param prompt_type: Prompt type to determine which checks to apply.
    :return: Tuple of (is_valid, cleaned_output).
    """
    cleaned = dict(output)

    if _has_entities(prompt_type) and "entities" in cleaned:
        if not cleaned["entities"]:
            return False, cleaned

    if _has_relations(prompt_type) and "relations" in cleaned:
        entity_names = {e["entity_name"] for e in cleaned.get("entities", [])}
        valid_relations = []
        for r in cleaned["relations"]:
            source = r.get("source_entity", "")
            target = r.get("target_entity", "")
            if source in entity_names and target in entity_names:
                valid_relations.append(r)
            else:
                logger.debug(
                    f"Dropping relation with invalid endpoints: "
                    f"{source} -> {target}"
                )
        removed = len(cleaned["relations"]) - len(valid_relations)
        if removed:
            logger.info(
                f"Removed {removed} relations with invalid endpoints "
                f"(kept {len(valid_relations)})"
            )
        cleaned["relations"] = valid_relations

    return True, cleaned


def _make_metadata(item: dict, language: str, now: str) -> dict:
    return {
        "domain": item["domain"],
        "difficulty": item["difficulty"],
        "language": item.get("language", language),
        "generated_at": now,
    }


async def synthesize_corpus(
    generator_llm: LLMOpenAI,
    config: dict,
    language: str,
) -> list[dict]:
    """
    Generate text corpus using large LLM.

    :param generator_llm: Generator LLM instance.
    :param config: Configuration dictionary.
    :param language: Target language.
    :return: List of corpus items with text, domain, difficulty.
    """
    corpus_config = config["corpus"]
    domains = corpus_config["domains"]
    difficulty_levels = corpus_config["difficulty_levels"]
    total_texts = corpus_config["total_texts"]
    min_text_length = config.get("quality_filters", {}).get("min_text_length", 80)

    texts_per_config = total_texts // (len(domains) * len(difficulty_levels))
    remainder = total_texts % (len(domains) * len(difficulty_levels))

    corpus = []
    count = 0

    logger.info(f"Synthesizing {total_texts} texts for language '{language}'")

    for domain in domains:
        for difficulty in difficulty_levels:
            num_texts = texts_per_config
            if count < remainder:
                num_texts += 1

            logger.info(
                f"Generating {num_texts} texts for domain '{domain}', "
                f"difficulty '{difficulty}'"
            )

            conversations = [
                [{"role": "user", "content": _build_synthesis_prompt(domain, difficulty, language)}]
                for _ in range(num_texts)
            ]

            domain_successes = 0
            try:
                texts = await generator_llm.batch_chat_completion(
                    conversations=conversations,
                    output_schema=str,
                    desc=f"Generating {domain}/{difficulty}",
                    temperature=generator_llm.kwargs.get("temperature", 0.2),
                )

                for text in texts:
                    text_clean = text.strip() if text else ""
                    if len(text_clean) < min_text_length:
                        logger.info(
                            f"  Text too short ({len(text_clean)} chars, "
                            f"min {min_text_length}), skipping: "
                            f"\"{_text_preview(text_clean)}\""
                        )
                        continue

                    corpus.append({
                        "text": text_clean,
                        "domain": domain,
                        "difficulty": difficulty,
                        "language": language,
                    })
                    count += 1
                    domain_successes += 1

            except Exception as e:
                logger.warning(
                    f"Failed to generate texts for {domain}/{difficulty}: {e}"
                )

            logger.info(
                f"Generated {domain_successes}/{num_texts} texts for "
                f"{domain}/{difficulty}"
            )

    logger.info(f"Synthesized {len(corpus)} texts for language '{language}'")
    return corpus


async def generate_examples_for_prompt_type(
    prompt_type: str,
    corpus: list[dict],
    generator_llm: LLMOpenAI,
    language: str,
    entity_types: list[str] | None = None,
    relation_types: list[str] | None = None,
) -> list[dict]:
    """
    Generate ICL examples using RAGU extractors for a specific prompt type.

    Instantiates the appropriate extractor and calls its internal batch
    methods, which use ``batch_chat_completion`` for concurrent LLM calls.

    :param prompt_type: One of artifact_extraction, artifact_validation,
        entity_extraction, entity_validation, relation_extraction,
        relation_validation.
    :param corpus: List of corpus items with text, domain, difficulty.
    :param generator_llm: Generator LLM instance.
    :param language: Target language.
    :param entity_types: Optional allowed entity types.
    :param relation_types: Optional allowed relation types.
    :return: List of example dicts ready for judging.
    """
    context = [item["text"] for item in corpus]
    now = datetime.now().isoformat() + "Z"

    try:
        if prompt_type == "artifact_extraction":
            return await _generate_artifact_extraction(
                context, corpus, generator_llm, language,
                entity_types, relation_types, now,
            )
        elif prompt_type == "artifact_validation":
            return await _generate_artifact_validation(
                context, corpus, generator_llm, language,
                entity_types, relation_types, now,
            )
        elif prompt_type == "entity_extraction":
            return await _generate_entity_extraction(
                context, corpus, generator_llm, language,
                entity_types, relation_types, now,
            )
        elif prompt_type == "entity_validation":
            return await _generate_entity_validation(
                context, corpus, generator_llm, language,
                entity_types, relation_types, now,
            )
        elif prompt_type == "relation_extraction":
            return await _generate_relation_extraction(
                context, corpus, generator_llm, language,
                entity_types, relation_types, now,
            )
        elif prompt_type == "relation_validation":
            return await _generate_relation_validation(
                context, corpus, generator_llm, language,
                entity_types, relation_types, now,
            )
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    except Exception as e:
        logger.warning(
            f"Failed to generate examples for {prompt_type}: {e}"
        )
        logger.debug(
            f"Error details: {type(e).__name__}: {e}", exc_info=True
        )
        return []


async def _generate_artifact_extraction(
    context: list[str],
    corpus: list[dict],
    generator_llm: LLMOpenAI,
    language: str,
    entity_types: list[str] | None,
    relation_types: list[str] | None,
    now: str,
) -> list[dict]:
    extractor = ArtifactsExtractorLLM(
        llm=generator_llm,
        do_validation=False,
        language=language,
        entity_types=entity_types,
        relation_types=relation_types,
    )
    results = await extractor._extract_artifacts(context)

    examples = []
    for item, model in zip(corpus, results):
        output = model.model_dump()
        is_valid, output = _clean_and_validate_output(output, "artifact_extraction")
        if not is_valid:
            logger.info(
                f"  Skipping: no entities for "
                f"\"{_text_preview(item['text'])}\""
            )
            continue
        examples.append({
            "input_text": item["text"],
            "metadata": _make_metadata(item, language, now),
            "output": output,
        })
    return examples


async def _generate_artifact_validation(
    context: list[str],
    corpus: list[dict],
    generator_llm: LLMOpenAI,
    language: str,
    entity_types: list[str] | None,
    relation_types: list[str] | None,
    now: str,
) -> list[dict]:
    extractor = ArtifactsExtractorLLM(
        llm=generator_llm,
        do_validation=True,
        language=language,
        entity_types=entity_types,
        relation_types=relation_types,
    )
    extracted = await extractor._extract_artifacts(context)
    results = await extractor._validate_artifacts(context, extracted)

    examples = []
    for item, model in zip(corpus, results):
        output = model.model_dump()
        is_valid, output = _clean_and_validate_output(
            output, "artifact_validation"
        )
        if not is_valid:
            logger.info(
                f"  Skipping: no entities for "
                f"\"{_text_preview(item['text'])}\""
            )
            continue
        examples.append({
            "input_text": item["text"],
            "metadata": _make_metadata(item, language, now),
            "output": output,
        })
    return examples


async def _generate_entity_extraction(
    context: list[str],
    corpus: list[dict],
    generator_llm: LLMOpenAI,
    language: str,
    entity_types: list[str] | None,
    relation_types: list[str] | None,
    now: str,
) -> list[dict]:
    extractor = TwoStageArtifactsExtractorLLM(
        llm=generator_llm,
        language=language,
        entity_types=entity_types,
        relation_types=relation_types,
    )
    results = await extractor._extract_entities(context)

    examples = []
    for item, model in zip(corpus, results):
        output = model.model_dump()
        output["relations"] = []
        is_valid, output = _clean_and_validate_output(
            output, "entity_extraction"
        )
        if not is_valid:
            logger.info(
                f"  Skipping: no entities for "
                f"\"{_text_preview(item['text'])}\""
            )
            continue
        examples.append({
            "input_text": item["text"],
            "metadata": _make_metadata(item, language, now),
            "output": output,
        })
    return examples


async def _generate_entity_validation(
    context: list[str],
    corpus: list[dict],
    generator_llm: LLMOpenAI,
    language: str,
    entity_types: list[str] | None,
    relation_types: list[str] | None,
    now: str,
) -> list[dict]:
    extractor = TwoStageArtifactsExtractorLLM(
        llm=generator_llm,
        do_entity_validation=True,
        language=language,
        entity_types=entity_types,
        relation_types=relation_types,
    )
    entity_models = await extractor._extract_entities(context)
    results = await extractor._validate_entities(context, entity_models)

    examples = []
    for item, model in zip(corpus, results):
        output = model.model_dump()
        output["relations"] = []
        is_valid, output = _clean_and_validate_output(
            output, "entity_validation"
        )
        if not is_valid:
            logger.info(
                f"  Skipping: no entities for "
                f"\"{_text_preview(item['text'])}\""
            )
            continue
        examples.append({
            "input_text": item["text"],
            "metadata": _make_metadata(item, language, now),
            "output": output,
        })
    return examples


async def _generate_relation_extraction(
    context: list[str],
    corpus: list[dict],
    generator_llm: LLMOpenAI,
    language: str,
    entity_types: list[str] | None,
    relation_types: list[str] | None,
    now: str,
) -> list[dict]:
    extractor = TwoStageArtifactsExtractorLLM(
        llm=generator_llm,
        language=language,
        entity_types=entity_types,
        relation_types=relation_types,
    )
    entity_models = await extractor._extract_entities(context)
    entities_payload = extractor._models_to_payload(entity_models)
    relation_models = await extractor._extract_relations(
        context, entities_payload
    )

    examples = []
    for item, ent_model, rel_model in zip(corpus, entity_models, relation_models):
        entities_payload = [e.model_dump() for e in ent_model.entities]
        output = rel_model.model_dump()
        output["entities"] = entities_payload
        is_valid, output = _clean_and_validate_output(
            output, "relation_extraction"
        )
        if not is_valid:
            logger.info(
                f"  Skipping: invalid output for "
                f"\"{_text_preview(item['text'])}\""
            )
            continue
        examples.append({
            "input_text": item["text"],
            "metadata": _make_metadata(item, language, now),
            "output": output,
            "entities": entities_payload,
        })
    return examples


async def _generate_relation_validation(
    context: list[str],
    corpus: list[dict],
    generator_llm: LLMOpenAI,
    language: str,
    entity_types: list[str] | None,
    relation_types: list[str] | None,
    now: str,
) -> list[dict]:
    extractor = TwoStageArtifactsExtractorLLM(
        llm=generator_llm,
        do_relation_validation=True,
        language=language,
        entity_types=entity_types,
        relation_types=relation_types,
    )
    entity_models = await extractor._extract_entities(context)
    entities_payload = extractor._models_to_payload(entity_models)
    relation_models = await extractor._extract_relations(
        context, entities_payload
    )
    validated = await extractor._validate_relations(
        context, entities_payload, relation_models
    )

    examples = []
    for item, ent_model, rel_model in zip(
        corpus, entity_models, validated
    ):
        entities_payload = [e.model_dump() for e in ent_model.entities]
        output = rel_model.model_dump()
        output["entities"] = entities_payload
        is_valid, output = _clean_and_validate_output(
            output, "relation_validation"
        )
        if not is_valid:
            logger.info(
                f"  Skipping: invalid output for "
                f"\"{_text_preview(item['text'])}\""
            )
            continue
        examples.append({
            "input_text": item["text"],
            "metadata": _make_metadata(item, language, now),
            "output": output,
            "entities": entities_payload,
        })
    return examples


async def judge_examples_batch(
    examples: list[dict],
    judge_llm: LLMOpenAI,
    config: dict,
    prompt_type: str,
) -> list[dict]:
    """
    Evaluate examples in batch using judge LLM with structured output.

    :param examples: List of example dicts with input_text and output.
    :param judge_llm: Judge LLM instance.
    :param config: Configuration dictionary.
    :param prompt_type: Prompt type to tailor evaluation criteria.
    :return: List of examples that passed quality threshold with rating.
    """
    if not examples:
        return []

    min_rating = config["judge_model"]["min_quality_rating"]
    judge_kwargs = {}
    if "anthropic" not in judge_llm.model_name.lower():
        judge_kwargs["temperature"] = 0.0

    conversations = []
    for example in examples:
        judge_prompt = _build_judge_prompt(example, prompt_type)
        conversations.append([{"role": "user", "content": judge_prompt}])

    try:
        ratings = await judge_llm.batch_chat_completion(
            conversations=conversations,
            output_schema=JudgeRatingModel,
            desc=f"Judging {prompt_type} examples",
            **judge_kwargs,
        )
    except Exception as e:
        logger.warning(f"Batch judging failed for {prompt_type}: {e}")
        return []

    accepted = []
    for example, rating_result in zip(examples, ratings):
        rating = getattr(rating_result, 'rating', None)
        if rating is None:
            logger.warning(
                f"  SKIPPED (no rating returned): "
                f"\"{_text_preview(example['input_text'])}\""
            )
            continue
        if rating >= min_rating:
            example["quality_rating"] = rating
            accepted.append(example)
            logger.info(
                f"  ACCEPTED (rating: {rating}/10): "
                f"\"{_text_preview(example['input_text'])}\""
            )
        else:
            logger.info(
                f"  REJECTED (rating: {rating}/10, min: {min_rating}): "
                f"\"{_text_preview(example['input_text'])}\""
            )

    logger.info(
        f"Judged {len(examples)} examples, {len(accepted)} accepted "
        f"(min rating: {min_rating})"
    )
    return accepted


def save_examples(
    examples: list[dict],
    output_file: str,
    config: dict,
) -> None:
    """
    Save examples to JSON file.

    :param examples: List of example dictionaries.
    :param output_file: Path to output JSON file.
    :param config: Configuration dictionary.
    """
    incremental = config["incremental"]

    existing_examples = []
    if incremental.get("enabled") and incremental.get("preserve_existing", True):
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_examples = existing_data.get("examples", [])

    new_examples_with_ids = []
    if incremental.get("generate_new_ids", True):
        for ex in examples:
            if "id" not in ex:
                ex["id"] = str(uuid4())
            new_examples_with_ids.append(ex)

    all_examples = existing_examples + new_examples_with_ids

    for ex in all_examples:
        if "id" not in ex:
            ex["id"] = str(uuid4())

    output_data = {
        "version": "1.0",
        "languages": config.get("languages", ["english", "russian"]),
        "total_examples": len(all_examples),
        "generated_by": {
            "generator_model": config["generator_model"]["model_name"],
            "judge_model": config["judge_model"]["model_name"],
        },
        "generated_at": datetime.now().isoformat() + "Z",
        "examples": all_examples,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Saved {len(all_examples)} examples to {output_file} "
        f"({len(new_examples_with_ids)} new, "
        f"{len(existing_examples)} existing)"
    )


async def main(
    config_path: str,
    language: str | None = None,
    input_texts_path: str | None = None,
) -> None:
    """
    Main entry point for ICL example generation.

    :param config_path: Path to YAML configuration file.
    :param language: Specific language to generate (None for all).
    :param input_texts_path: Path to custom texts (None to synthesize).
    """
    config = load_config(config_path)

    logger.info("Initializing generator LLMs...")
    gen_base_url = config["generator_model"]["base_url"]
    gen_api_key = config["generator_model"]["api_key"]
    gen_model_name = config["generator_model"]["model_name"]
    gen_temperature = config["generator_model"].get("temperature", 0.2)

    synthesis_llm = LLMOpenAI(
        client=CachedAsyncOpenAI(base_url=gen_base_url, api_key=gen_api_key),
        model_name=gen_model_name,
        temperature=gen_temperature,
    )

    extraction_llm = LLMOpenAI(
        client=CachedAsyncOpenAI(base_url=gen_base_url, api_key=gen_api_key, cache={}),
        model_name=gen_model_name,
        temperature=gen_temperature,
    )

    logger.info("Initializing judge LLM...")
    judge_kwargs = {}
    if "anthropic" not in config["judge_model"]["model_name"].lower():
        judge_kwargs["temperature"] = 0.0

    judge_llm = LLMOpenAI(
        client=CachedAsyncOpenAI(
            base_url=config["judge_model"]["base_url"],
            api_key=config["judge_model"]["api_key"],
            cache={},
        ),
        model_name=config["judge_model"]["model_name"],
        **judge_kwargs,
    )

    languages = [language] if language else config["languages"]

    for lang in languages:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing language: {lang}")
        logger.info(f"{'=' * 60}\n")

        texts_path = input_texts_path or config.get("input_texts", {}).get("path")
        if texts_path:
            logger.info(f"Loading texts from {texts_path}")
            corpus = list(filter(lambda it: it["language"] == lang, load_input_texts(texts_path, languages)))
        else:
            corpus = await synthesize_corpus(synthesis_llm, config, lang)

        corpus = deduplicate_corpus(corpus)

        if not corpus:
            logger.warning(f"No texts available for language '{lang}', skipping")
            continue

        entity_types = config.get("entity_types")
        relation_types = config.get("relation_types")

        for prompt_type in config["prompt_types"]:
            logger.info(f"\nGenerating examples for: {prompt_type}")

            existing_texts = load_existing_texts(config["output_path"], prompt_type)
            corpus_for_type = [
                it for it in corpus
                if it["text"].strip() not in existing_texts
            ]

            if not corpus_for_type:
                logger.info(
                    f"All {len(corpus)} texts already processed for "
                    f"{prompt_type}, skipping"
                )
                continue

            skipped = len(corpus) - len(corpus_for_type)
            if skipped:
                logger.info(
                    f"Skipping {skipped} texts already in "
                    f"{prompt_type}_examples.json, "
                    f"processing {len(corpus_for_type)}"
                )

            examples = await generate_examples_for_prompt_type(
                prompt_type=prompt_type,
                corpus=corpus_for_type,
                generator_llm=extraction_llm,
                language=lang,
                entity_types=entity_types,
                relation_types=relation_types,
            )

            if not examples:
                logger.warning(f"No valid examples generated for {prompt_type}")
                continue

            accepted = await judge_examples_batch(
                examples=examples,
                judge_llm=judge_llm,
                config=config,
                prompt_type=prompt_type,
            )

            if accepted:
                output_file = os.path.join(
                    config["output_path"],
                    f"{prompt_type}_examples.json",
                )
                save_examples(accepted, output_file, config)
            else:
                logger.warning(f"No examples passed judge for {prompt_type}")

    logger.info("\nExample generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate in-context learning examples for RAGU extractors"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/icl_generation.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["english", "russian"],
        help="Generate examples for specific language only",
    )
    parser.add_argument(
        "--input-texts",
        type=str,
        default=None,
        help="Path to custom texts (directory of .txt, .json array, or single .txt)",
    )

    args = parser.parse_args()

    asyncio.run(main(args.config, args.language, args.input_texts))
