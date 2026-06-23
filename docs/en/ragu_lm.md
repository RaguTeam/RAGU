# RAGU-lm

RAGU-lm is a compact fine-tuned model (Qwen-3-0.6B) specialized for structured
information extraction from Russian-language text. It is trained on the NEREL
dataset and is used through the `RaguLmArtifactExtractor` adapter.

> RAGU-lm is intended for **Russian-language** corpora. For other languages use
> the LLM-based extractors (`ArtifactsExtractorLLM`, `TwoStageArtifactsExtractorLLM`).
> See [`ragu_components.md`](ragu_components.md) for an overview and
> [`../../ragu/triplet/README.md`](../../ragu/triplet/README.md) for the API.

The model is trained to perform four tasks:

1. Recognize unnormalized named entities from text.
2. Normalize each unnormalized entity with respect to the given text.
3. Generate a definition for each normalized named entity based on the text.
4. Generate a definition of the directed relation between a pair of normalized
   named entities within the text.

---

## Prompts

**System prompt:**

```
Вы - эксперт в области анализа текстов и извлечения семантической информации из них.
```

**1. Entity extraction:**

```
Распознайте все именованные сущности в тексте и выпишите их список с новой строки.\n\nТекст: {input_text}\n\nИменованные сущности:
```

**2. Normalization:**

```
Выполните нормализацию именованной сущности, встретившейся в тексте.\n\nИсходная (ненормализованная) именованная сущность: {source_entity}\n\nТекст: {source_text}\n\nНормализованная именованная сущность:
```

**3. Entity description generation:**

```
Напишите, что означает именованная сущность в тексте, то есть раскройте её смысл относительно текста.\n\nИменованная сущность: {normalized_entity}\n\nТекст: {source_text}\n\nСмысл именованной сущности:
```

**4. Relation extraction:**

```
Напишите, что означает отношение между двумя именованными сущностями в тексте, то есть раскройте смысл этого отношения относительно текста (либо напишите прочерк, если между двумя именованными сущностями отсутствует отношение).\n\nПервая именованная сущность: {first_normalized_entity}\n\nВторая именованная сущность: {second_normalized_entity}\n\nТекст: {source_text}\n\nСмысл отношения между двумя именованными сущностями:
```

---

## Worked example

The four prompts below operate on the **same source text** (a news snippet about
the Obama family buying a house in Washington). The input text is shown once,
then each task lists only its specific input and expected output.

<details>
<summary><b>Source text shared by all four tasks</b></summary>

Семья Обамы приобрела дом в Вашингтоне за 8,1 млн долларов Барак Обама Бывший президент США Барак Обама с женой Мишель приобрели жильё в Вашингтоне недалеко от Белого дома. После окончания срока работы на посту президента в январе супруги арендовали особняк в стиле эпохи Тюдоров в престижном районе столицы США — Калорама, где долгое время селились дипломаты, лоббисты и политики. Младшей дочери Барака и Мишель — Саше осталось ещё два года до окончания частной школы в Вашингтоне. Поэтому семья решила выкупить дом с восемью спальными комнатами стоимостью 8,1 миллиона долларов. Ранее дом принадлежал бывшему пресс-секретарю экс-президента США Билла Клинтона Джо Локхарту (), который в настоящее время возглавляет пресс-службу Национальной футбольной лиги США. Среди соседей Обамы — видные деятели Вашингтона, включая дочь действующего президента Иванку Трамп и её мужа Джареда Кушнера (оба являются советниками Белого дома), которые переехали в район Калорама из Нью-Йорка в начале 2017 года. Госсекретарь США Рекс Тиллерсон тоже живёт в доме неподалеку. «Учитывая, что президент и миссис Обама планируют жить в Вашингтоне как минимум ещё два года, для них имело смысл приобрести дом в собственность, а не продолжать его арендовать», — сообщил представитель Обамы. Семья президента Обамы также владеет домом в Чикаго — третьем по величине городе в США, где Обама приобрёл значительную политическую поддержку перед тем, как дважды стал главою Белого дома.

</details>

### Task 1 — Entity extraction

**Input:** the source text above (with the entity-extraction prompt).

**Expected output:**

```text
Семья Обамы
Обамы
приобрела дом
Вашингтоне
8,1 млн долларов
Барак Обама
Обама
президент США
президент
США
Мишель
приобрели жильё
Белого дома
президента
в январе
Тюдоров
столицы США
Калорама
дипломаты
лоббисты
политики
Барака
Саше
два года
восемью
8,1 миллиона долларов
пресс-секретарю экс-президента США
пресс-секретарю
президента США
Билла Клинтона
Джо Локхарту
пресс-службу Национальной футбольной лиги США
Национальной футбольной лиги США
Вашингтона
Иванку Трамп
Джареда Кушнера
советниками Белого дома
Нью-Йорка
в начале 2017 года
Госсекретарь США
Госсекретарь
Рекс Тиллерсон
миссис Обама
представитель Обамы
Семья президента Обамы
Чикаго
третьем
главою Белого дома
```

### Task 2 — Normalization

**Input:** `Исходная (ненормализованная) именованная сущность: пресс-секретарю` + source text.

**Expected output:**

```text
пресс-секретарь
```

### Task 3 — Entity description generation

**Input:** `Именованная сущность: пресс-секретарь` + source text.

**Expected output:**

```text
Бывший представитель СМИ экс-президента США Билла Клинтона.
```

### Task 4 — Relation extraction

**Input:**

- `Первая именованная сущность: Джо Локхарт`
- `Вторая именованная сущность: пресс-служба Национальной футбольной лиги США`
- source text.

**Expected output:**

```text
Возглавляет пресс-службу Национальной футбольной лиги США.
```

---

## Comparison

| Model                 | Dataset | F1 (Entities) | F1 (Relations) |
|-----------------------|---------|---------------|----------------|
| Qwen-2.5-14B-Instruct | NEREL   | 0.32          | 0.69           |
| RAGU-lm (Qwen-3-0.6B) | NEREL   | 0.60          | 0.71           |
| Small-model pipeline  | NEREL   | 0.74          | 0.75           |

---

## How to use

It is recommended to serve RAGU-lm with [vLLM](https://docs.vllm.ai/en/latest/).

Start the server:

```bash
vllm serve RaguTeam/ragu-lm --max_model_len 4096
```

Initialize `RaguLmArtifactExtractor` and use it like any other extractor:

```python
from ragu.chunker.types import Chunk
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet.ragu_lm_artifact_extractor import RaguLmArtifactExtractor

client = CachedAsyncOpenAI(
    base_url="http://0.0.0.0:8000/v1/",
    api_key="dummy-api-token",
)
llm = LLMOpenAI(
    client=client,
    model_name="RaguTeam/ragu-lm",
)

pipeline = RaguLmArtifactExtractor(
    llm=llm,
)

chunks = [Chunk(content="Some source text.", chunk_order_idx=0, doc_id="doc-1")]
entities, relations = await pipeline.extract(chunks)
```
