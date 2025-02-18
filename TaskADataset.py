from torch.utils.data import Dataset, DataLoader

class TaskADataset(Dataset):
    def __init__(self, examples, labels):
        self.examples = examples
        self.labels = labels
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "text": self.examples[idx],
            "label": self.labels[idx]
        }
