import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNeg(nn.Module):

    def __init__(self, vocab_size, emb_dim):

        super().__init__()

        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)

        init_range = 0.5 / emb_dim

        nn.init.uniform_(self.in_embed.weight, -init_range, init_range )

        nn.init.zeros_(self.out_embed.weight)

    def forward(self, input_words, target_words, negative_words):

        inp_vec = self.in_embed(input_words)

        target_vec = self.out_embed(target_words)

        neg_vec = self.out_embed(negative_words)

        pos_score = torch.sum(inp_vec * target_vec, dim=1)

        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm( neg_vec,inp_vec.unsqueeze(2))

        neg_score = neg_score.squeeze(2)

        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        total_loss = -(pos_loss + neg_loss)

        return total_loss.mean()