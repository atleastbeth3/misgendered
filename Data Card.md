# MISGENDERED Dataset

## Contact
Tamanna Hossain: tthossai@uci.edu

## Authors
- Tamanna Hossain, UCI
- Sunipa Dev, Google Research
- Sameer Singh, UCI

## Paper
[MISGENDERED: Limits of Large Language Models in Understanding Pronouns](https://arxiv.org/abs/2306.03950)
(ACL 2023)

## Description of Content
The MISGENDERED dataset consists of templates (`templates/`) for <i>gendering</i> individuals given a set of their preferred pronouns. Each template consists of a slot for an individual’s `{name}` and slots for explicit declarations of their pronouns (`{nom}`, `{acc}`, `{pos_dep}`, `{pos_ind}`, `{ref}`), followed by a sentence in which the model has to predict a missing `[PRONOUN]`. 

The declared and missing pronouns cover 5 different pronoun forms: nominative, accusative, possessive-dependent, possessive-independent, and reflexive for 11 sets of pronouns (`pronouns.csv`) from 3 pronoun types: binary (e.g., he, she), gender-neutral (e.g., they, them), and neo-pronouns (e.g., xe, thon).

300 unisex names sampled from [Flowers (2015)](https://github.com/fivethirtyeight/data/tree/master/unisex-names), and top 100 names associated with male or female in the US ([Social Security, 2022](https://www.ssa.gov/oact/babynames/decades/century.html)) are used to populate the template.

In total the dataset consists of total of 3.3 million instances: 10 templates for each of the 5 pronoun forms with 13 declaration types, 11 pronoun sets, and 500 names.

## Intended Use Case
- Evaluate LLMs on their understanding of pronouns

## Unsuitable Use Cases
- As a benchmark for ensuring fairness or lack of fairness
- As a resource for bias mitigation in production systems

## Prohibited Use Cases
- <b>Harmful Actions:</b> The dataset should not be used for any purpose or in any manner that promotes or facilitates harm, discrimination, or negative biases against non-binary individuals or any other group.
- <b>Targeting and Profiling:</b> The dataset should not be used to target, profile, or single out individuals based on their gender identity or any other personal attributes.