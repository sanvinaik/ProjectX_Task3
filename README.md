1. Structure of Repo

TASK 3/
├── PAPER_NOTES.md    # Reading notes (claim, core architecture, evaluation)
├── README.md         # Instructions, dependencies, and expectations
├── text8             # Clean Wikipedia text dataset file (Unzipped)
└── src/              
    ├── data.py       # Word2VecDataset class with Subsampling & Negative Sampling
    ├── model.py      # SkipGramNeg PyTorch Neural Network module
    └── train.py      # Main execution script (Training Loop & Evaluation)

2. Dataset is not uploaded because it was a 100 mb file so it was not getting uploaded to github.
http://mattmahoney.net/dc/text8.zip
Here is the link of the dataset, I trained my model on it only and removed it later while uploading to github.

3. Parameters 
EMBED_DIM = 100
BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 0.025
WINDOW_SIZE = 2
NEG_SAMPLES = 5

4. To run the code 
python src/train.py

5. Expected output-

Epoch 1: Loss = 4.8210
Epoch 2: Loss = 3.9452
Epoch 3: Loss = 3.2104
Epoch 4: Loss = 2.6540
Epoch 5: Loss = 2.1105

Similarity Check
one - two : 0.6421
king - queen : 0.4102
american - french : 0.5294

But results are varying because of only first 10k words being used.
 