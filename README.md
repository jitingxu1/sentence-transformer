# Task 1
A Sentence Transformer takes an input sentence (or sentences) and encodes it into a fixed-dimensional embedding that captures the semantic meaning of the sentence. While under the hood, we often use a pretrained Transformer (e.g., BERT, RoBERTa, DistilBERT), we also need a pooling strategy to convert the final hidden states into a single vector.

Typical steps:

Pass the tokenized sentence through the Transformer.
Extract the hidden states.
Aggregate/pool them (e.g., CLS token, average pooling, max pooling) to get a single sentence-level embedding

---

# Task 2
## 1. High-Level Overview of Multi-Task Learning
In multi-task learning, you share a common backbone (basemodel) while having separate task-specific heads. For this sentence transformer:
  - Shared Backbone
      - Pretrained Transformer (e.g., BERT, DistilBERT) for encoding text into contextual embeddings.
      - Potentially a pooling layer to produce sentence embeddings.
  - Task-Specific Heads
      - Task A (Sentence Classification): Usually a simple feed-forward layer on top of the pooled embedding (e.g., [CLS]).
      - Task B Named Entity Recognition (NER): Requires token-level classification (one label per token).

Because these tasks have different label spaces and sometimes different input/output shapes (especially if one task is sentence-level and another is token-level), we design separate heads while reusing the same backbone.

---

## 2. Changes to the Architecture
### 2.1 Shared Encoder (Backbone + Pooling)
1. Transformer Encoder (e.g., bert-base-uncased)
    - Converts tokenized input into hidden states.
2. Pooling Layer (optional, depending on the task)
    - For sentence-level tasks, you might use [CLS] pooling or average pooling.
    - For token-level tasks, you often use the entire hidden state sequence without an additional pooling layer.

### 2.2 Task A Head: Sentence Classification
- Input: A pooled sentence embedding (shape: [batch_size, hidden_dim]).
- Architecture:
    - Dense layer(s) (e.g., nn.Linear(hidden_dim, hidden_dim/2) + activation).
    - Output classification layer (e.g., nn.Linear(hidden_dim/2, num_classes)), with a softmax or cross-entropy loss.
- Loss: Typically cross-entropy for classification.


### 2.3 Task B Head
Depends on the nature of the second task:

Case 1: Named Entity Recognition (NER)
- Input: The full sequence of hidden states from the Transformer (shape: [batch_size, seq_len, hidden_dim]).
- Architecture:
    - Possibly a CRF layer on top or a simple nn.Linear(hidden_dim, num_labels) for each token.
    - If using a CRF, you’ll need a separate decoding mechanism.
- Loss:
    - Token-level cross-entropy (each token predicted independently) or CRF-based loss.

### 2.4 Shared vs. Independent Parameters
- Backbone: Shared across tasks.
- Heads: Independent parameters for each task.
- Potential to share early layers of the heads if tasks are similar, but typically each head is distinct.


---

# Task 3: Training Considerations

Below is a comprehensive overview of **training considerations** for a multi-task Sentence Transformer with **two tasks** (Task A: Sentence Classification, Task B: e.g., NER or Sentiment Analysis). 
We’ll first discuss the pros, cons, and implications of freezing different parts of the network, 
then move into how to apply **transfer learning** effectively, including which layers to freeze/unfreeze and why.

---

## 1. Freezing vs. Fine-Tuning Different Parts of the Network

### 1.1 Freezing the Entire Network

- **Scenario**  
  You use a **pretrained Sentence Transformer** (or a pretrained backbone + pooling) and **do not update any parameters** during training.
  Instead, you might only do a forward pass to **extract embeddings** for your tasks.

- **Advantages**  
  1. **Speed and Simplicity**: No backprop through the network → minimal compute cost.  
  2. **Avoids Overfitting** if your dataset for the new tasks is very small.

- **Drawbacks**  
  1. **No Task Adaptation**: The model won’t learn new domain-specific patterns.  
  2. **Potentially Suboptimal Performance**: Especially if your tasks differ significantly from the original pretraining domain.

- **When to Consider**  
  - **Tiny new dataset** where fine-tuning could lead to overfitting.  
  - Rapid prototyping or embedding extraction, where **performance is secondary to speed**.

---

### 1.2 Freezing Only the Transformer Backbone

- **Scenario**  
  You keep the **pretrained Transformer** weights fixed and **train/update only the task-specific heads** on top of the frozen backbone.

- **Advantages**  
  1. **Reduced Computation**: Fine-tuning fewer parameters is faster and typically less memory-intensive.  
  2. **Stability**: Reduces the risk of catastrophic forgetting or overfitting the backbone on small task data.  
  3. **Task-Specific Adaptation**: The classification/NER heads still learn from the backbone features but adapt to your tasks.

- **Drawbacks**  
  1. **Limited Adaptation at the Encoder Level**: If your tasks differ from the backbone’s pretraining domain, a frozen backbone might not capture all relevant nuances.  
  2. **Moderate Performance Gains**: Typically better than freezing everything, but not always optimal if the domain is unique.

- **When to Consider**  
  - You have **medium-sized** data for your tasks.  
  - You want a **balance** between computational efficiency and performance.  
  - The tasks are **somewhat similar** to the domain of the pretrained model.

---

### 1.3 Freezing Only One Task-Specific Head

- **Scenario**  
  Suppose you have two heads:  
  - **Task A Head** (fully trainable)  
  - **Task B Head** (frozen)  
  The **backbone** can be fully trainable or partially frozen, but **one** head remains untouched.

- **Advantages**  
  1. **Protect Well-Performing Task**: If one task is already performing well and you don’t want to degrade it, freezing its head preserves its performance.  
  2. **Focus on the New/Underperforming Task**: The other task head can be fine-tuned to catch up.

- **Drawbacks**  
  1. **Inconsistent Knowledge Sharing**: If the frozen head cannot adapt to changes in the backbone, you risk **misalignment**. As the backbone updates to help Task A, Task B might degrade if it’s frozen.  
  2. **Maintenance Complexity**: You need to be sure the frozen head is stable under the changes from training the rest of the network.

- **When to Consider**  
  - You already have a **well-validated** head for one task and want to **introduce or improve** another task.  
  - You’re certain the changes in the backbone for Task A won’t hurt Task B performance (this can be tricky to guarantee).

---

## 2. Transfer Learning Process

### 2.1 Choice of a Pre-trained Model

1. **Domain Similarity**  
   Pick a backbone (e.g., `bert-base-uncased`, `roberta-base`, or domain-specific variants like `clinicalBERT` if you’re dealing with clinical data).  
2. **Size vs. Performance**  
   - If you have **limited compute** or need faster inference, consider a smaller model like `distilbert-base-uncased`.  
   - For **best performance** on general text, a larger model like `roberta-large` may be better.

**Example Rationale**  
- If your tasks involve **general English text**, `bert-base-uncased` is a strong all-purpose choice.  
- If your tasks involve **financial text** or **medical text**, look for specialized models (e.g., FinBERT, BioBERT).

---

### 2.2 Layers to Freeze/Unfreeze

**Recommended Approach**  
1. **Unfreeze the Task Heads**: Always train the classification or NER heads to adapt to your tasks.  
2. **Partially Unfreeze the Transformer**:  
   - **Option A**: Unfreeze the last _N_ layers of the backbone.  
   - **Option B**: Unfreeze the entire Transformer if you have enough data and computational resources.

**Why Partial Freezing?**  
- It reduces the risk of losing general knowledge from earlier Transformer layers, which capture **universal linguistic patterns**.  
- Letting the final layers adapt helps with **domain/task-specific patterns**.

---

### 2.3 Rationale for These Choices

1. **Task-Specific Adaptation**  
   Your classification/NER tasks likely differ from the original pretraining task (e.g., masked language modeling). Fine-tuning the last layers and heads improves performance on specific label sets.

2. **Prevent Overfitting**  
   Freezing earlier layers maintains **general linguistic features**. If you unfreeze everything with a small dataset, you could overfit or even degrade the model’s general language capabilities.

3. **Computational Efficiency**  
   Fine-tuning only some Transformer layers plus the heads is **faster** and uses **less memory** than fully fine-tuning a large model.

4. **Leverage Pretrained Knowledge**  
   Pretrained models capture **rich semantic representations**. By partially unfreezing, you can **retain** these representations while still **adapting** to your tasks.

---


# Task 4: Training Loop Implementation (BONUS)

This section outlines how a **training loop** might be structured for the **Multi-Task Learning (MTL)** scenario described in Task 2. We focus on **conceptual explanations** rather than actual code, highlighting:

- **Hypothetical data handling**  
- **Forward pass logic**  
- **Metrics** for evaluating each task  

> **Note:** We do not actually train the model here. 

---

## 1. Handling Hypothetical Data

1. **Multiple Datasets**  
   - You typically have **one dataset per task**.  
   - **Task A (Sentence Classification)** might consist of `(sentence, label)` pairs, where each label is an integer class (e.g., 0, 1, 2).  
   - **Task B (Token Classification)** might have `(sentence, [token_labels])`, where `[token_labels]` aligns with the tokens in the sentence.

2. **Combining Data or Interleaving**  
   - For **true multi-task learning**, you can **interleave** batches of Task A and Task B in the same training loop.  
   - Alternatively, you can **sample** from each task in proportion to the size or importance of that task.

3. **Batch Construction**  
   - Each batch (for each task) should contain the **input text** and **task-specific labels** (e.g., integer class labels for classification, token labels for NER).  
   - You also need a way to **distinguish which task** a given batch belongs to, if you’re using a single loop that processes batches from different tasks.

4. **Key Assumption**:  
   - There is **no real-world conflict** in mixing these datasets, and each dataset is **sufficiently large** or **representative** for the tasks.

---

## 2. Forward Pass Considerations

1. **Shared Encoder**  
   - You have a **pretrained transformer** that converts tokens into contextualized embeddings.  
   - This encoder is shared by **both Task A and Task B**.

2. **Separate Heads**  
   - **Task A Head** (Sentence Classification):  
     - Applies to the **pooled** encoder output (e.g., the `[CLS]` token or average of all tokens).  
     - Produces **logits** of shape `(batch_size, num_classes_taskA)`.
   - **Task B Head** (Token Classification):  
     - Applies to the **full sequence** of embeddings (token-level).  
     - Produces **logits** of shape `(batch_size, seq_len, num_labels_taskB)`.

3. **Selecting the Right Head**  
   - During training, the loop must specify **which head** to use based on the **task**.  
   - You might store a “task” key in each batch to tell the model which **forward logic** to follow.

4. **Key Assumption**:  
   - The model can handle varying shapes (sentence-level vs. token-level) by switching to the correct output head without conflict.

---

## 3. Metrics

1. **Task-Specific Evaluation**  
   - **Task A (Classification)**: Typically uses **accuracy**, **precision**, **recall**, **F1**, etc., at the **sentence level**.  
   - **Task B (Token-Level)**: Often needs **token-level accuracy** or an **F1 score** (especially for NER with BIO tags).

2. **Periodic Assessment**  
   - After some batches (or at the end of each epoch), evaluate **both tasks** on their respective validation sets.  
   - Keeps track of performance so that one task doesn’t degrade while optimizing for the other.

3. **Combined or Separate Metrics**  
   - You can track **separate metrics** for each task, or consider a **joint metric** if tasks share an overarching goal.  
   - Typically, separate metrics are easier to interpret.

4. **Key Assumption**:  
   - Each task has a **reliable evaluation metric** and **sufficient validation data** to measure progress accurately.

---

## 4. Final Thoughts

By **conceptually** interleaving Task A and Task B within one training loop, you:
- **Share** the transformer encoder, learning general linguistic patterns.  
- **Specialize** via separate heads for distinct label spaces (classification vs. token-level tagging).  
- **Track** separate metrics, ensuring neither task degrades while focusing on the other.

> A real-world implementation would require **careful batching** (especially for token-level tasks) and **robust evaluation** to confirm that **both tasks** benefit from multi-task learning.

---

### **Key Takeaways**
- **MTL Setup**: Use a **shared encoder** with **task-specific heads**.  
- **Data Handling**: Keep tasks separate but interleave them in a **single loop**.  
- **Forward Pass**: Decide **which head** to run based on the current task.  
- **Metrics**: Maintain **separate metrics** to ensure each task is improving.  
- **Loss and Updates**: Optionally **weight losses** or **alternate** tasks to balance learning.


