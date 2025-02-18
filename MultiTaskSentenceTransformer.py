import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes_taskA=3, num_labels_taskB=5):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Task A: Sentence Classification head
        hidden_dim = self.encoder.config.hidden_size
        self.taskA_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes_taskA)
        )
        
        # Task B: NER head (token-level classification)
        self.taskB_classifier = nn.Linear(hidden_dim, num_labels_taskB)
    
    def forward(self, input_texts, task="A", labels=None):
        # 1. Tokenize
        encoding = self.tokenizer(
            input_texts, padding=True, truncation=True, return_tensors="pt"
        )
        
        # 2. Encoder Forward
        outputs = self.encoder(**encoding)
        
        if task == "A":
            # Sentence classification
            # Using [CLS] token (index 0) for demonstration
            cls_emb = outputs.last_hidden_state[:, 0, :]
            logits = self.taskA_classifier(cls_emb)
            
            return logits  # (batch_size, num_classes_taskA)
        
        elif task == "B":
            # NER: token-level classification
            sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            logits = self.taskB_classifier(sequence_output)
            
            # logits shape: (batch_size, seq_len, num_labels_taskB)
            return logits
        
        else:
            raise ValueError("Invalid task specification. Use 'A' or 'B'.")


# Example usage
if __name__ == "__main__":
    model = MultiTaskSentenceTransformer(num_classes_taskA=3, num_labels_taskB=5)
    
    # Task A: Sentence Classification
    sample_sentences = [
        "I love pizza!",
        "The new movie was terrible.",
    ]
    taskA_logits = model(sample_sentences, task="A")
    print("Task A logits shape:", taskA_logits.shape)  # (batch_size, num_classes_taskA)
    
    # Task B: NER
    sample_sentences_ner = [
        "Barack Obama was the 44th President of the United States.",
        "Apple is looking at buying a startup in the UK."
    ]
    taskB_logits = model(sample_sentences_ner, task="B")
    print("Task B logits shape:", taskB_logits.shape)  # (batch_size, seq_len, num_labels_taskB)
