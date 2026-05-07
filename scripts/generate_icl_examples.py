"""
Generate in-context learning examples for RAGU extractors.

This script synthesizes a corpus of texts, generates artifacts
using a large LLM (generator), evaluates quality using another
large LLM (judge), and saves examples to JSON files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List
from uuid import uuid4

import yaml


sys.path.insert(0, str(Path(__file__).parent.parent))

from ragu.common.logger import logger
from ragu.common.prompts.messages import render as render_messages
from ragu.common.prompts.prompt_storage import DEFAULT_PROMPT_TEMPLATES
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI

from ragu.triplet.prompts import (
    TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION,
    TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION,
    TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION,
    TWO_STAGE_RELATION_VALIDATION_INSTRUCTION,
)


PROMPT_INSTRUCTIONS = {
    "artifact_extraction": DEFAULT_PROMPT_TEMPLATES["artifact_extraction"],
    "artifact_validation": DEFAULT_PROMPT_TEMPLATES["artifact_validation"],
    "entity_extraction": TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION,
    "entity_validation": TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION,
    "relation_extraction": TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION,
    "relation_validation": TWO_STAGE_RELATION_VALIDATION_INSTRUCTION,
}


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
    language: str
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


def _build_judge_prompt(
    example: dict,
    prompt_type: str,
) -> str:
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
        f"   - {e.get('entity_name', 'Unknown')} ({e.get('entity_type', 'Unknown')}): {e.get('description', 'No description')[:80]}"
        for e in output.get("entities", [])
    ])

    relations_list = "\n".join([
        f"   - {r.get('source_entity', 'Unknown')} → {r.get('target_entity', 'Unknown')} ({r.get('relation_type', 'Unknown')}): {r.get('description', 'No description')[:80]}"
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
        "1. Accuracy: Are all extracted items correct and grounded in the text? Check each claim against the source text. Penalize hallucinations — items not supported by the text.",
        "2. Completeness: Did it miss obvious items that should have been extracted from the text?",
        "3. Quality: Are types, descriptions, and other fields appropriate and informative?",
    ]
    if has_entities:
        criteria_lines.append(
            f"4. Entities: Are entity names properly normalized (capitalized, canonical form)? "
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
        data_lines.append(f"\n**Extracted Entities ({entity_count}):**\n{entities_list}")
    if has_relations:
        data_lines.append(f"\n**Extracted Relations ({relation_count}):**\n{relations_list}")
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

Rate the overall quality from 1 to 10 based on the criteria above.

Provide your answer in the following format:
Rating: [1-10]
Explanation: [Brief 1-2 sentence explanation]
"""
    return prompt.strip()


def _extract_rating_from_text(text: str) -> int | None:
    """
    Extract numeric rating from LLM response.

    :param text: LLM response text.
    :return: Numeric rating or None.
    """
    patterns = [
        r"Rating:\s*(\d+)",
        r"(\d+)\s*/\s*10",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                rating_str = match.group(1)
                rating = int(rating_str)
                if 1 <= rating <= 10:
                    logger.debug(f"Extracted rating: {rating}")
                    return rating
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse rating from '{match.group(1)}': {e}")
                continue

    logger.warning(f"Could not extract rating from text: {text[:200]}")
    return None


async def _call_llm(
    instruction,
    generator_llm: LLMOpenAI,
    **render_kwargs,
):
    """
    Render instruction template and call generator LLM.

    All Jinja2 templates use StrictUndefined, so optional parameters
    (examples, entity_types, relation_types) must be explicitly passed
    as None when not provided.

    :param instruction: RAGUInstruction to use.
    :param generator_llm: Generator LLM instance.
    :param render_kwargs: Keyword arguments passed to render().
    :return: Pydantic model instance from LLM response.
    """
    defaults = {
        "examples": None,
        "entity_types": None,
        "relation_types": None,
    }
    for k, v in defaults.items():
        render_kwargs.setdefault(k, v)

    convs = render_messages(instruction.messages, **render_kwargs)
    result = await generator_llm.chat_completion(
        conversation=convs[0].to_openai(),
        output_schema=instruction.pydantic_model,
        temperature=generator_llm.kwargs.get("temperature", 0.2),
    )
    return result


async def synthesize_corpus(
    generator_llm: LLMOpenAI,
    config: dict,
    language: str
) -> List[dict]:
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

            domain_successes = 0

            for i in range(num_texts):
                synthesis_prompt = _build_synthesis_prompt(
                    domain=domain,
                    difficulty=difficulty,
                    language=language
                )

                try:
                    text = await generator_llm.chat_completion(
                        conversation=[
                            {"role": "user", "content": synthesis_prompt}
                        ],
                        output_schema=str,
                        temperature=generator_llm.kwargs.get("temperature", 0.2)
                    )

                    text_clean = text.strip()
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
                        "language": language
                    })
                    count += 1
                    domain_successes += 1
                    logger.debug(
                        f"  Generated text ({len(text_clean)} chars): "
                        f"\"{_text_preview(text_clean)}\""
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to generate text for {domain}/{difficulty}: {e}"
                    )

            logger.info(
                f"Generated {domain_successes}/{num_texts} texts for "
                f"{domain}/{difficulty}"
            )

    logger.info(
        f"Synthesized {len(corpus)} texts for language '{language}'"
    )
    return corpus


async def generate_artifacts(
    text: str,
    prompt_type: str,
    generator_llm: LLMOpenAI,
    language: str
) -> tuple[dict, dict]:
    """
    Extract artifacts from text using generator LLM.

    For multi-stage prompt types (relation extraction/validation), performs
    prerequisite extraction steps automatically before calling the target prompt.

    :param text: Input text.
    :param prompt_type: Type of prompt (entity_extraction, etc.).
    :param generator_llm: Generator LLM instance.
    :param language: Target language.
    :return: Tuple of (output_dict, context_dict) where output_dict contains
             the extraction result and context_dict contains additional data
             needed for example structure (e.g. entities for relation prompts).
    """
    instruction = PROMPT_INSTRUCTIONS[prompt_type]

    if prompt_type == "artifact_extraction":
        result = await _call_llm(
            instruction, generator_llm,
            context=[text], language=language,
        )
        return result.model_dump(), {}

    elif prompt_type == "entity_extraction":
        result = await _call_llm(
            instruction, generator_llm,
            context=[text], language=language,
        )
        return result.model_dump(), {}

    elif prompt_type == "relation_extraction":
        entity_result = await _call_llm(
            TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION, generator_llm,
            context=[text], language=language,
        )
        entities_payload = [e.model_dump() for e in entity_result.entities]
        result = await _call_llm(
            instruction, generator_llm,
            context=[text], entities=[entities_payload], language=language,
        )
        output = result.model_dump()
        output["entities"] = entities_payload
        return output, {"entities": entities_payload}

    elif prompt_type == "artifact_validation":
        extract_result = await _call_llm(
            DEFAULT_PROMPT_TEMPLATES["artifact_extraction"], generator_llm,
            context=[text], language=language,
        )
        result = await _call_llm(
            instruction, generator_llm,
            context=[text], artifacts=[extract_result], language=language,
        )
        return result.model_dump(), {}

    elif prompt_type == "entity_validation":
        entity_result = await _call_llm(
            TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION, generator_llm,
            context=[text], language=language,
        )
        entities_payload = [e.model_dump() for e in entity_result.entities]
        result = await _call_llm(
            instruction, generator_llm,
            context=[text], entities=[entities_payload], language=language,
        )
        return result.model_dump(), {}

    elif prompt_type == "relation_validation":
        entity_result = await _call_llm(
            TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION, generator_llm,
            context=[text], language=language,
        )
        entities_payload = [e.model_dump() for e in entity_result.entities]

        relation_result = await _call_llm(
            TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION, generator_llm,
            context=[text], entities=[entities_payload], language=language,
        )
        relations_payload = [r.model_dump() for r in relation_result.relations]

        result = await _call_llm(
            instruction, generator_llm,
            context=[text], entities=[entities_payload],
            relations=[relations_payload], language=language,
        )
        output = result.model_dump()
        output["entities"] = entities_payload
        return output, {"entities": entities_payload}

    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def _text_preview(text: str, max_len: int = 120) -> str:
    """Return a truncated single-line preview of text for logging."""
    single = text.replace("\n", " ")
    if len(single) > max_len:
        return single[:max_len] + "..."
    return single


def _cleanup_artifacts(artifacts: dict, prompt_type: str) -> dict:
    """
    Remove invalid entries from LLM-generated artifacts.

    Strips entities missing required fields and relations whose endpoints
    are not in the entity list. Modifies the dict in place.

    :param artifacts: Output dict with 'entities' and/or 'relations'.
    :param prompt_type: Prompt type to determine which sections to clean.
    :return: Cleaned artifacts dict.
    """
    has_entities = prompt_type in (
        "artifact_extraction", "artifact_validation",
        "entity_extraction", "entity_validation",
        "relation_extraction", "relation_validation",
    )
    has_relations = prompt_type in (
        "artifact_extraction", "artifact_validation",
        "relation_extraction", "relation_validation",
    )

    if has_entities and "entities" in artifacts:
        cleaned = []
        for e in artifacts["entities"]:
            if not e.get("entity_name") or not e.get("entity_type"):
                logger.debug(f"Dropping entity without name/type: {e}")
                continue
            if not e.get("description"):
                logger.debug(f"Dropping entity without description: {e.get('entity_name')}")
                continue
            cleaned.append(e)
        removed = len(artifacts["entities"]) - len(cleaned)
        if removed:
            logger.debug(f"Removed {removed} invalid entities")
        artifacts["entities"] = cleaned

    if has_relations and "relations" in artifacts:
        entity_names = {e["entity_name"] for e in artifacts.get("entities", [])}
        cleaned = []
        for r in artifacts["relations"]:
            source = r.get("source_entity", "")
            target = r.get("target_entity", "")
            if source in entity_names and target in entity_names:
                cleaned.append(r)
            else:
                logger.debug(
                    f"Dropping relation with invalid endpoints: "
                    f"{source} -> {target}"
                )
        removed = len(artifacts["relations"]) - len(cleaned)
        if removed:
            logger.info(
                f"Removed {removed} relations with invalid endpoints "
                f"(kept {len(cleaned)})"
            )
        artifacts["relations"] = cleaned

    return artifacts


async def judge_example(
    example: dict,
    judge_llm: LLMOpenAI,
    config: dict,
    prompt_type: str,
) -> tuple[bool, int | None]:
    """
    Evaluate example quality using judge LLM.

    :param example: Example dictionary with input_text and output.
    :param judge_llm: Judge LLM instance.
    :param config: Configuration dictionary.
    :param prompt_type: Prompt type to tailor evaluation criteria.
    :return: Tuple of (is_good, quality_rating).
    """
    judge_prompt = _build_judge_prompt(example, prompt_type)

    try:
        llm_kwargs = {"output_schema": str}
        if "anthropic" not in judge_llm.model_name.lower():
            llm_kwargs["temperature"] = 0.0

        rating_text = await judge_llm.chat_completion(
            conversation=[{"role": "user", "content": judge_prompt}],
            **llm_kwargs,
        )

        rating = _extract_rating_from_text(rating_text)
        min_rating = config["judge_model"]["min_quality_rating"]

        is_good = rating is not None and rating >= min_rating
        return is_good, rating

    except Exception as e:
        logger.warning(f"Failed to judge example: {e}")
        return False, None


def validate_example(
    example: dict,
    prompt_type: str,
) -> tuple[bool, str]:
    """
    Validate example for semantic correctness.

    Checks that extracted data is structurally complete and internally
    consistent. Does NOT enforce arbitrary limits on text length or
    entity/relation counts — those are soft recommendations for the
    synthesis stage, not quality gates.

    :param example: Example dictionary.
    :param prompt_type: Prompt type to determine which checks to apply.
    :return: Tuple of (is_valid, reason). reason is empty string on success.
    """
    output = example["output"]

    has_entities = prompt_type in (
        "artifact_extraction", "artifact_validation",
        "entity_extraction", "entity_validation",
        "relation_extraction", "relation_validation",
    )
    has_relations = prompt_type in (
        "artifact_extraction", "artifact_validation",
        "relation_extraction", "relation_validation",
    )

    if has_entities:
        entities = output.get("entities", [])
        if not entities:
            return False, "No entities extracted"

        for entity in entities:
            if not entity.get("entity_name") or not entity.get("entity_type"):
                return False, (
                    f"Entity missing name or type: {entity}"
                )
            if not entity.get("description"):
                return False, (
                    f"Entity missing description: {entity.get('entity_name')}"
                )

    if has_relations:
        entity_names = {e["entity_name"] for e in output.get("entities", [])}
        for relation in output.get("relations", []):
            source = relation.get("source_entity")
            target = relation.get("target_entity")
            if source not in entity_names or target not in entity_names:
                return False, (
                    f"Relation endpoints not in entity list: "
                    f"{source} -> {target}"
                )

    return True, ""


def save_examples(
    examples: List[dict],
    output_file: str,
    config: dict
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
            "judge_model": config["judge_model"]["model_name"]
        },
        "generated_at": datetime.now().isoformat() + "Z",
        "examples": all_examples
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Saved {len(all_examples)} examples to {output_file} "
        f"({len(new_examples_with_ids)} new, "
        f"{len(existing_examples)} existing)"
    )


async def main(config_path: str, language: str | None = None) -> None:
    """
    Main entry point for ICL example generation.

    :param config_path: Path to YAML configuration file.
    :param language: Specific language to generate (None for all).
    """
    config = load_config(config_path)

    logger.info("Initializing generator LLM...")
    generator_llm = LLMOpenAI(
        client=CachedAsyncOpenAI(
            base_url=config["generator_model"]["base_url"],
            api_key=config["generator_model"]["api_key"],
        ),
        model_name=config["generator_model"]["model_name"],
        temperature=config["generator_model"].get("temperature", 0.2)
    )

    logger.info("Initializing judge LLM...")
    judge_kwargs = {}
    if "anthropic" not in config["judge_model"]["model_name"].lower():
        judge_kwargs["temperature"] = 0.0

    judge_llm = LLMOpenAI(
        client=CachedAsyncOpenAI(
            base_url=config["judge_model"]["base_url"],
            api_key=config["judge_model"]["api_key"],
        ),
        model_name=config["judge_model"]["model_name"],
        **judge_kwargs
    )

    languages = [language] if language else config["languages"]

    for lang in languages:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing language: {lang}")
        logger.info(f"{'='*60}\n")

        corpus = await synthesize_corpus(
            generator_llm, config, lang
        )

        for prompt_type in config["prompt_types"]:
            logger.info(f"\nGenerating examples for: {prompt_type}")

            all_examples = []
            processed_count = 0

            for idx, item in enumerate(corpus, 1):
                text_preview = _text_preview(item["text"])
                logger.info(
                    f"[{idx}/{len(corpus)}] Processing "
                    f"{item['domain']}/{item['difficulty']}: "
                    f"\"{text_preview}\""
                )
                try:
                    artifacts, extra_context = await generate_artifacts(
                        text=item["text"],
                        prompt_type=prompt_type,
                        generator_llm=generator_llm,
                        language=lang
                    )

                    n_ent = len(artifacts.get("entities", []))
                    n_rel = len(artifacts.get("relations", []))
                    logger.info(
                        f"  Generated: {n_ent} entities, {n_rel} relations"
                    )

                    example = {
                        "id": str(uuid4()) if config["incremental"].get(
                            "generate_new_ids", True
                        ) else None,
                        "input_text": item["text"],
                        "metadata": {
                            "domain": item["domain"],
                            "difficulty": item["difficulty"],
                            "language": lang,
                            "generated_at": datetime.now().isoformat() + "Z"
                        },
                        "output": artifacts,
                    }

                    if "entities" in extra_context:
                        example["entities"] = extra_context["entities"]

                    _cleanup_artifacts(example["output"], prompt_type)

                    n_ent_a = len(example["output"].get("entities", []))
                    n_rel_a = len(example["output"].get("relations", []))
                    if n_ent_a != n_ent or n_rel_a != n_rel:
                        logger.info(
                            f"  After cleanup: {n_ent_a} entities, "
                            f"{n_rel_a} relations"
                        )

                    valid, reason = validate_example(example, prompt_type)
                    if not valid:
                        logger.info(
                            f"  REJECTED by validation: {reason}\n"
                            f"  Text: \"{text_preview}\""
                        )
                        continue

                    is_good, rating = await judge_example(
                        example, judge_llm, config, prompt_type
                    )

                    if is_good and rating is not None:
                        example["quality_rating"] = rating
                        all_examples.append(example)
                        logger.info(
                            f"  ACCEPTED (rating: {rating}/10)"
                        )
                        processed_count += 1
                    else:
                        logger.info(
                            f"  REJECTED by judge (rating: {rating}/10, "
                            f"min: {config['judge_model']['min_quality_rating']})\n"
                            f"  Text: \"{text_preview}\""
                        )

                except Exception as e:
                    logger.warning(
                        f"  ERROR processing {item['domain']}/"
                        f"{item['difficulty']}: {e}\n"
                        f"  Text: \"{text_preview}\""
                    )
                    logger.debug(f"Error details: {type(e).__name__}: {e}", exc_info=True)

            logger.info(
                f"Processed {processed_count}/{len(corpus)} texts successfully, "
                f"generated {len(all_examples)} valid examples"
            )

            if all_examples:
                output_file = os.path.join(
                    config["output_path"],
                    f"{prompt_type}_examples.json"
                )
                save_examples(all_examples, output_file, config)
            else:
                logger.warning(
                    f"No valid examples generated for {prompt_type}"
                )

    logger.info("\nExample generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate in-context learning examples for RAGU extractors"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/icl_generation.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["english", "russian"],
        help="Generate examples for specific language only"
    )

    args = parser.parse_args()

    asyncio.run(main(args.config, args.language))
