import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

class SentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', pooling='mean'):
        super().__init__()
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
    
    def forward(self, sentences):
        # Tokenize input
        inputs = self.tokenizer(sentences, return_tensors='pt', 
                                padding=True, truncation=True)
        
        # Model outputs
        outputs = self.transformer(**inputs)
        
        # The last hidden states have shape [batch_size, seq_len, hidden_dim]
        last_hidden_states = outputs.last_hidden_state
        
        # Pooling methods: CLS, mean, todo: max
        if self.pooling == 'cls':
            # CLS token is at index 0
            sentence_embeddings = last_hidden_states[:, 0, :]
        elif self.pooling == 'mean':
            # Mean of words embeddings
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
        else:
            raise ValueError("Invalid pooling method")
        
        return sentence_embeddings
