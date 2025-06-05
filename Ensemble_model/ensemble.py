import torch

class BotEnsemble:
    def __init__(self, transformer_model, tokenizer, numeric_model, alpha=0.5, device='cuda'):
        self.transformer = transformer_model.eval().to(device)
        self.tokenizer = tokenizer
        self.numeric_model = numeric_model.eval().to(device)
        self.alpha = alpha
        self.device = device

    def predict_prob(self, userdesc, features):
        with torch.no_grad():
            feats_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            mlp_prob = torch.sigmoid(self.numeric_model(feats_tensor)).item()

        if userdesc and userdesc.strip():
            inputs = self.tokenizer(userdesc, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.transformer(**inputs).logits
                transformer_prob = torch.sigmoid(logits).item()
            final_prob = self.alpha * transformer_prob + (1 - self.alpha) * mlp_prob
        else:
            final_prob = mlp_prob

        return final_prob
