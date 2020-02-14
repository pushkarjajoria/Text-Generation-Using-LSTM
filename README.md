# Text-Prediction-Using-LSTM

Tensorflow 1

I used the book "The Count of Monte Cristo" from https://www.gutenberg.org

The input is of dimensions (total_blocks x 16 x 256 x k)
where the block size is 16 with 256 characters in each block, one-hot encoded
for 'k' most frequent characters in the book.

The model contains 2 LSTM cells with 256 units each and a softmax output layer
with k units.