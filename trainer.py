import torch
import torch.nn.functional as F

model = MultiTaskSentenceTransformer(
    model_name="bert-base-uncased", 
    num_classes_taskA=3, 
    num_labels_taskB=5
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# For metrics
def accuracy(predictions, labels):
    preds = torch.argmax(predictions, dim=-1)
    return (preds == labels).float().mean()

num_epochs = 2

# We'll assume these loaders are small. In a real scenario, you'd want more robust iteration logic.
taskA_iter = iter(taskA_loader)
taskB_iter = iter(taskB_loader)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")

    # Reset/reenable iterators at each epoch
    taskA_iter = iter(taskA_loader)
    taskB_iter = iter(taskB_loader)

    while True:
        try:
            # ---- Task A Step ----
            batchA = next(taskA_iter)
            textA = batchA["text"]
            labelsA = batchA["label"]

            optimizer.zero_grad()
            logitsA = model(textA, task="A")
            lossA = F.cross_entropy(logitsA, labelsA)
            lossA.backward()
            optimizer.step()

            # For demonstration: compute accuracy
            accA = accuracy(logitsA, labelsA)
            print(f"[Task A] Loss: {lossA.item():.4f}, Acc: {accA.item():.2f}")

            # ---- Task B Step ----
            batchB = next(taskB_iter)
            textB = batchB["text"]
            labelsB = batchB["ner_labels"]

            optimizer.zero_grad()
            logitsB = model(textB, task="B")
            
            # logitsB: (batch_size, seq_len, num_labels)
            # labelsB: needs to match (batch_size, seq_len) shape
            # We'll do a simple token-level cross-entropy
            # We'll flatten them for simplicity

            batch_size, seq_len, num_labels = logitsB.shape
            logitsB_flat = logitsB.view(-1, num_labels)
            labelsB_flat = torch.tensor(labelsB).view(-1)
            lossB = F.cross_entropy(logitsB_flat, labelsB_flat)

            lossB.backward()
            optimizer.step()

            # NER metrics can be more involved (F1 score, entity-level)
            # For demonstration, we'll do a quick "accuracy" check
            predsB_flat = torch.argmax(logitsB_flat, dim=-1)
            accB = (predsB_flat == labelsB_flat).float().mean()
            print(f"[Task B] Loss: {lossB.item():.4f}, Token Acc: {accB.item():.2f}")

        except StopIteration:
            # One of the loaders ran out of data
            break
