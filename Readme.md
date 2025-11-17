# sugardata

`sugardata` is a Python package that helps you **generate synthetic datasets for NLP tasks**, enabling experimentation, prototyping, and training when labeled data is limited or unavailable.

![](media/logo.png)

# Installation

You can install **sugardata** via pip:

```bash
pip install sugardata
```

# Quick Start

`sugardata` supports multiple NLP tasks out of the box.

## Task 1: Sentiment Analysis

```python

import sugardata as su

# generate synchronously
results = su.generate_sentiment_data(concept="online shopping")

# generate asynchronously
results = await su.generate_sentiment_data_async(concept="online shopping")

# generate concurrently with multiple vendors
vendors = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash-lite",
    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    "ollama": "gemma3:12b",
}
results = await su.generate_sentiment_multi_vendor_async(
    concept="online shopping",
    vendors=vendors
)

# augment synchronously
examples = [
    # some text
    ...
]

results = su.augment_sentiment_data(examples=examples)

# augment asynchronously
results = await su.augment_sentiment_data_async(examples=examples)

# augment concurrently with multiple vendors
vendors = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash-lite",
    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    "ollama": "gemma3:12b",
}
results = await su.augment_sentiment_multi_vendor_async(
    examples=examples,
    vendors=vendors
)

```

## Task 2: Named Entity Recognition (NER) Localization

```python

import sugardata as su

examples = [
    {
        "text": "John works at Acme Corp in New York",
        "ner_tags": [{"John": "PER", "Acme Corp": "ORG", "New York": "LOC"}]
    },
    # more examples...
]

entity_labels = {"PER": (1, 2), "ORG": (3, 4), "LOC": (5, 6)}

# localize synchronously
results = su.localize_ner_data(
    examples=examples,
    language="Turkish",
    entity_labels=entity_labels
)

# localize asynchronously
results = await su.localize_ner_data_async(
    examples=examples,
    language="Turkish",
    entity_labels=entity_labels
)

# localize concurrently with multiple vendors
vendors = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash-lite",
    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    "ollama": "gemma3:12b",
}
results = await su.localize_ner_data_multi_vendor_async(
    examples=examples,
    language="Turkish",
    entity_labels=entity_labels,
    vendors=vendors
)

```

To learn more about configuration options, advanced parameters, and integration tips, please visit tutorials.
