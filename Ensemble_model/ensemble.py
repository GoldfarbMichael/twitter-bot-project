import torch
from torch.nn.functional import softmax
import pandas as pd
class BotEnsemble:
    def __init__(self, transformer_model, tokenizer, numeric_model, alpha=0.5, device='cuda'):
        self.transformer = transformer_model.eval().to(device)
        self.tokenizer = tokenizer
        self.numeric_model = numeric_model.eval().to(device)
        self.alpha = alpha
        self.device = device

    def predict_prob(self, acctdesc, features):
        feats_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mlp_prob = torch.sigmoid(self.numeric_model(feats_tensor)).item()

        # If acctdesc is available, use transformer too
        if acctdesc and not pd.isna(acctdesc) and str(acctdesc).strip():
            inputs = self.tokenizer(
                acctdesc,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                return_token_type_ids=False
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.transformer(**inputs).logits
                probs = softmax(logits, dim=1)
                transformer_prob = probs[0, 1].item()  # class 1 = bot

            final_prob = self.alpha * transformer_prob + (1 - self.alpha) * mlp_prob
        else:
            final_prob = mlp_prob

        return final_prob

    def predict_label(self, acctdesc, features, threshold=0.5):

        prob = self.predict_prob(acctdesc, features)
        return int(prob > threshold)




