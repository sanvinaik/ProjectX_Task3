import math
import random
import torch
from torch.utils.data import Dataset


class Word2VecDataset(Dataset):

# defining winodw size and no of neg samples
    def __init__(self, text, window_size=2, num_neg_samples=5, subsample_thresh=1e-3):
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples

#sepearting the words, converted them to lower case and split into list
        words = text.lower().split()
        
        w_count = {}
        for word in words:
            if word in w_count:
                w_count[word] = w_count[word] + 1
            else:
                w_count[word] = 1

        wc_list = []
        for word in w_count:
            wc_list.append((word, w_count[word]))


     # put words in tuple and put it in a list and index of list sort according to frequency in descending order and we get the word id     
        wc_list.sort(key=lambda x: x[1], reverse=True)

        self.i2w = []
        for pair in wc_list:
            self.i2w.append(pair[0])

        self.w2i = {}
        cnt = 0
        for word in self.i2w:
            self.w2i[word] = cnt
            cnt = cnt + 1
        #total size of vocablury

        self.vocab_size = len(self.i2w)
        total_words = len(words)

# rarer the word higher the prob, if score is greater than 1 prop is 1 else prob is prob
        keep_prob = {}
        for word in w_count:
            count = w_count[word]
            freq = count / total_words
            z = freq / subsample_thresh
            
            prob = (math.sqrt(z) + 1) / z
            if prob > 1:
                prob = 1
                
            keep_prob[word] = prob

            #now we draw random no b/w 0 and 1 and if number is smaller than prob it goes in filtered

        filtered = []
        for word in words:
            rand_num = random.random()
            if rand_num < keep_prob[word]:
                filtered.append(word)


        #filteretd word ids get
        word_ids = []
        for word in filtered:
            word_ids.append(self.w2i[word])

        # now to chose random wrong words power the count of each word to 0.75
        weights = []
        for i in range(self.vocab_size):
            word = self.i2w[i]
            count = w_count[word]
            weights.append(count ** 0.75)

        # divide by total to get prob
        total_sum = 0
        for w in weights:
            total_sum = total_sum + w

        self.neg_probs = []
        for w in weights:
            self.neg_probs.append(w / total_sum)

        self.pairs = []
        for i in range(len(word_ids)):
            center = word_ids[i]

            left = i - self.window_size
            if left < 0:
                left = 0
                
            right = i + self.window_size + 1
            if right > len(word_ids):
                right = len(word_ids)

            for j in range(left, right):
                if i == j:
                    continue
                context = word_ids[j]
                self.pairs.append((center, context))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]

        vocab_ids = []
        for i in range(self.vocab_size):
            vocab_ids.append(i)

        neg_words = random.choices(
            vocab_ids,
            weights=self.neg_probs,
            k=self.num_neg_samples
        )

        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(neg_words, dtype=torch.long)
        )