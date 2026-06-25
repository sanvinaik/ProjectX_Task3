Paper- Efficient Estimation of Word Representations in Vector Space (Word2Vec)

1. Central Claim-

The paper begins with how computationally expensive training neural networks is on very big data sets like billions of words. So if we remove the non linear hidden layer from the network models can be trained way faster than the time they took before. They worked one one hot encoding. Also old models didn't capture relationship between words, words having similar meanings were placed at the same distance words not having any smililarity were. So we will use a multi dimensional vecotr space. 

2. The Core Arhcitecture or Algorithm-

The paper has two log linear methods- Continuous Bag of Words and Continuous Skip Gram

* CBOW- The input will be a sentence with a missing center word and the model will find that missing word

* Skip Gram- A central word is given as input and the model predicts its sorrounding words. 

I have used Skip Gram and to further optimize it I used it with subsampling and negative sampling.

* Subsampling- Removing words like the, a, this that appear too many times to optimize.

* Negative Sampling- Instead of comparing all the words, for every correct word pair, the model selets wrong random words using probabality function , in which high score is give to correct word pair and low to wrong pair.

3. Dataset and Evaluation 

The data set actually used was Google news corpus but it is not public so text 8 data is used which is public and also provided in the paper.

I am only feeding the first 10K words of even the 100 mb data because it takes a lot of time otherwise to cpmpute.

Then I have done a total of 5 epocs, basically going over the data 5 times, and average loss of these 5 epochs are printed, and every time loss is reducing which means similar words are getting closer to each other on the vector space.
Also just like the paper has done at last, took 3 pairs of words to give them a score to see how accurate the model is.
The dataset is quite small so it isnt that accurate but its still good for 10k words only.



