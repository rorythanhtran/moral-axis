# Task overview

You are helping classify personality trait differentials according to Moral Foundations Theory.

Use the provided foundation definitions and dictionary as your primary reference framework. The goal is to apply the Moral Foundations framework consistently.

## Rating columns

Rate each trait differential for:

- `care`
- `fairness`
- `loyalty`
- `authority`
- `purity`
- `liberty`
- `general`

Use `general` for broad moral goodness or badness, virtue or vice, righteousness, immorality, ethical character, or similar general moral meaning that does not fit neatly into one specific foundation.

## Scale

For each category, assign:

- `1` if the trait differential is conceptually related to the category.
- `0` if the trait differential is not conceptually related to the category.

Use only `1` or `0` for rating values.

## Instructions

- Rate whether the trait pair is related to the foundation, not whether the trait is morally good or bad.
- If either pole is clearly related to a foundation, mark that foundation as `1`.
- Rate each trait differential independently. Do not infer a broader personality story from the broader trait list.
- If only one pole motivates the rating, mention that briefly in `notes`.
- Keep notes short. Include a brief reason only when it helps explain an ambiguous rating or which pole matched.
- Return one completed rating object for every input trait.
- Do not invent new traits or change trait indices.
- Do not include explanations outside the requested JSON.

## Required JSON Shape

Return exactly one JSON object with this shape:

```json
{
  "ratings": [
    {
      "trait_index": 1,
      "differential": "playful :: serious",
      "left_pole": "playful",
      "right_pole": "serious",
      "care": 0,
      "fairness": 0,
      "loyalty": 0,
      "authority": 0,
      "purity": 0,
      "liberty": 0,
      "general": 0,
      "notes": ""
    }
  ]
}
```

## MFT Definitions

{foundation_definitions}

## MFT Dictionary

{foundation_dictionary}

## Traits To Rate

{batch_tsv}
