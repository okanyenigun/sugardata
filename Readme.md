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

# generate concurrently with multiple vendor
results = await su.generate_sentiment_multi_vendor_async(concept="online shopping")

# augment synchronously
examples = [
    # some text
    ...
]

results = su.augment_sentiment_data(examples=examples)

# augment asynchronously
results = await su.augment_sentiment_data_async(examples=examples)

# augment concurrently with multiple vendor
results = await su.augment_sentiment_multi_vendor_async(examples=examples)

```

To learn more about configuration options, advanced parameters, and integration tips, please visit tutorials.
