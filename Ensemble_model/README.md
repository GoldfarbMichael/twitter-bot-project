# ensemble.py

The `BotEnsemble` class combines predictions from two models:
- A **transformer-based text classifier** (e.g. DistilBERT) on the user's bio/description.
- A **numeric metadata model** (e.g. sklearn, PyTorch, or ONNX) using features like retweets and followers.

Final prediction = weighted average of the two models.

---

### Dependencies

```python
import torch
from torch.nn.functional import softmax
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import onnxruntime as ort
```
Class: BotEnsemble

```python
BotEnsemble(transformer_model, tokenizer, numeric_model, alpha=0.5, device='cuda')
```
### Parameters:
Parameter	Type	Description
transformer_model	torch.nn.Module	Pretrained transformer model (e.g. BERT) for bios
tokenizer	HuggingFace tokenizer	Tokenizer for the transformer model
numeric_model	sklearn / PyTorch / ONNX	Model that predicts from numeric metadata
alpha	float (default=0.5)	Weight given to transformer model in the final prediction
device	'cuda' or 'cpu'	Device to run the models on

### Methods

```python
predict_prob(acctdesc, features)
```
Returns the bot probability as a float between 0 and 1.

```python
prob = model.predict_prob("crypto investor & engineer", [avg_retweetcount, followers])
```
If acctdesc is valid, uses both transformer and numeric model.

Final probability:

```python
final_prob = alpha * transformer_prob + (1 - alpha) * numeric_prob
predict_label(acctdesc, features, threshold=0.5)
```
Returns a binary label: 1 for bot, 0 for human.

```python
label = model.predict_label("crypto investor", [250, 120])
```

Model Type	Description:
- PyTorch	Any torch.nn.Module
- Sklearn	Any BaseEstimator with predict_proba
- ONNX	InferenceSession using ONNX runtime

### Internal Workflow
Numeric Model Prediction

### PyTorch:
```python
torch.sigmoid(model(tensor_input)).item()
```
### Sklearn:

python
Copy
Edit
model.predict_proba(df)[0][1]
ONNX:

```python
outputs = model.run(None, {input_name: np_array})
prob = outputs[1][0][1]
```
Transformer Model Prediction

Uses HuggingFace tokenizer + model

Computes:

```python
softmax(logits, dim=1)[0, 1].item()
```
### Final Output:

- Weighted combination:

```python
final_prob = alpha * transformer_prob + (1 - alpha) * numeric_prob
```
### Error Handling
Raises a TypeError if numeric_model is not one of:

torch.nn.Module

sklearn.BaseEstimator

onnxruntime.InferenceSession

### Example Usage
```python
ensemble = BotEnsemble(
    transformer_model=bert_model,
    tokenizer=bert_tokenizer,
    numeric_model=sklearn_model,
    alpha=0.6,
    device='cuda'
)

acctdesc = "🇷🇺 Patriot, crypto analyst, anti-fake news 💯"
features = [320, 1500]  # [avg_retweetcount, followers]

print(ensemble.predict_prob(acctdesc, features))  # → 0.76
print(ensemble.predict_label(acctdesc, features))  # → 1
```
### Notes
- The ensemble allows combining text and metadata for better classification.

- It's flexible to different numeric model types.

- Useful in bot detection, fake profile classification, and hybrid ML systems.
