from torch.utils.data import Dataset, DataLoader


class TaskBDataset(Dataset):
    def __init__(self, examples, ner_labels):
        self.examples = examples
        self.ner_labels = ner_labels
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "text": self.examples[idx],
            "ner_labels": self.ner_labels[idx]  # token-level IDs
        }
