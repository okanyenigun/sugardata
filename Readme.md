# sugardata

`sugardata` is a Python package that helps you **generate synthetic datasets for NLP tasks**, enabling experimentation, prototyping, and training when labeled data is limited or unavailable.

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

results = su.generate_sentiments(concept="online shopping", n_samples=100)

```

## Task 2: Aspect-Based Sentiment Analysis (ABSA)

```python

import sugardata as su

results = su.generate_aspect_sentiments(concept="smartphones", aspects=["battery life", "camera", "price"], n_samples=100)

```

To learn more about configuration options, advanced parameters, and integration tips, please visit tutorials.
