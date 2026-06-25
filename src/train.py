import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import Word2VecDataset
from model import SkipGramNeg


EMBED_DIM = 100
BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 0.025
WINDOW_SIZE = 2
NEG_SAMPLES = 5


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
text8_path = os.path.join(project_root, "text8")

with open(text8_path, "r", encoding="utf-8") as file:
    text = file.read()
    real_text = " ".join(text.split()[:10000])

dataset = Word2VecDataset(real_text,window_size=WINDOW_SIZE,num_neg_samples=NEG_SAMPLES)

dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

model = SkipGramNeg( dataset.vocab_size, EMBED_DIM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = optim.AdamW(model.parameters(),lr=LEARNING_RATE)

for epoch in range(EPOCHS):

    total_loss = 0

    for centers, contexts, negatives in dataloader:

        centers = centers.to(device)
        contexts = contexts.to(device)
        negatives = negatives.to(device)

        if negatives.dim() > 2:
            negatives = negatives.squeeze(1)

        optimizer.zero_grad()

        loss = model(centers, contexts, negatives)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

word_matrix = model.in_embed.weight.data


def get_similarity(word1, word2):

    if word1 not in dataset.w2i:
        return f"'{word1}' missing from vocabulary"

    if word2 not in dataset.w2i:
        return f"'{word2}' missing from vocabulary"

    idx1 = dataset.w2i[word1]
    idx2 = dataset.w2i[word2]

    vec1 = word_matrix[idx1]
    vec2 = word_matrix[idx2]

    sim = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))

    return f"{sim.item():.4f}"


print("\nSimilarity Check")
print(f"one - two : {get_similarity('one', 'two')}")
print(f"king - queen : {get_similarity('king', 'queen')}")
print(f"american - french : {get_similarity('american', 'french')}")