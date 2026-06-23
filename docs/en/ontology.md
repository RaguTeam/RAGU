# NEREL Ontology

RAGU uses the entity and relation classes from [NEREL](https://github.com/nerel-ds/NEREL) as its default ontology. The NEREL type lists are injected into the extraction prompts out of the box.

To use a different ontology, pass custom `entity_types` and `relation_types` to any LLM extractor (`ArtifactsExtractorLLM` or `TwoStageArtifactsExtractorLLM`). See [`ragu/triplet/README.md`](../../ragu/triplet/README.md) for the constructor signature.

The canonical type lists also live in code as `NEREL_ENTITY_TYPES` and `NEREL_RELATION_TYPES` in `ragu/triplet/types.py`.

---

## Entity types

| No. | Entity type | No. | Entity type  | No. | Entity type   |
|-----|-------------|-----|--------------|-----|---------------|
| 1.  | AGE         | 11. | FAMILY       | 21. | PENALTY       |
| 2.  | AWARD       | 12. | IDEOLOGY     | 22. | PERCENT       |
| 3.  | CITY        | 13. | LANGUAGE     | 23. | PERSON        |
| 4.  | COUNTRY     | 14. | LAW          | 24. | PRODUCT       |
| 5.  | CRIME       | 15. | LOCATION     | 25. | PROFESSION    |
| 6.  | DATE        | 16. | MONEY        | 26. | RELIGION      |
| 7.  | DISEASE     | 17. | NATIONALITY  | 27. | STATE_OR_PROV |
| 8.  | DISTRICT    | 18. | NUMBER       | 28. | TIME          |
| 9.  | EVENT       | 19. | ORDINAL      | 29. | WORK_OF_ART   |
| 10. | FACILITY    | 20. | ORGANIZATION |     |               |

## Relation types

| No. | Relation type    | No. | Relation type      | No. | Relation type    |
|-----|------------------|-----|--------------------|-----|------------------|
| 1.  | ABBREVIATION     | 18. | HEADQUARTERED_IN   | 35. | PLACE_RESIDES_IN |
| 2.  | AGE_DIED_AT      | 19. | IDEOLOGY_OF        | 36. | POINT_IN_TIME    |
| 3.  | AGE_IS           | 20. | INANIMATE_INVOLVED | 37. | PRICE_OF         |
| 4.  | AGENT            | 21. | INCOME             | 38. | PRODUCES         |
| 5.  | ALTERNATIVE_NAME | 22. | KNOWS              | 39. | RELATIVE         |
| 6.  | AWARDED_WITH     | 23. | LOCATED_IN         | 40. | RELIGION_OF      |
| 7.  | CAUSE_OF_DEATH   | 24. | MEDICAL_CONDITION  | 41. | SCHOOLS_ATTENDED |
| 8.  | CONVICTED_OF     | 25. | MEMBER_OF          | 42. | SIBLING          |
| 9.  | DATE_DEFUNCT_IN  | 26. | ORGANIZES          | 43. | SPOUSE           |
| 10. | DATE_FOUNDED_IN  | 27. | ORIGINS_FROM       | 44. | START_TIME       |
| 11. | DATE_OF_BIRTH    | 28. | OWNER_OF           | 45. | SUBEVENT_OF      |
| 12. | DATE_OF_CREATION | 29. | PARENT_OF          | 46. | SUBORDINATE_OF   |
| 13. | DATE_OF_DEATH    | 30. | PART_OF            | 47. | TAKES_PLACE_IN   |
| 14. | END_TIME         | 31. | PARTICIPANT_IN     | 48. | WORKPLACE        |
| 15. | EXPENDITURE      | 32. | PENALIZED_AS       | 49. | WORKS_AS         |
| 16. | FOUNDED_BY       | 33. | PLACE_OF_BIRTH     |     |                  |
| 17. | HAS_CAUSE        | 34. | PLACE_OF_DEATH     |     |                  |
