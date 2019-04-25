4 classes classification: archaeal, bacterial, fungi and plant tRNA sequences.

Data: train/valid/test = 8000/1000/3000 total samples

Layers count includes normalization and activation, without input layer

Approaches (on the same data):
  
1. Image data 
   * original sequences of different length, parsing-provided images were resized to 80x80
   * acc = 0.98, loss = 0.04, val_acc = 0.94, val_loss = 0.27
   * 200 epochs
   * 20 layers
   * test results: true = 2796, false = 204
   * test accuracy = 93.2%
2. Vectorized data
   * origial sequences length = 220, parsing-provided uint32 vectors to bytes
   * acc = 0.97, loss = 0.06, val_acc = 0.94, val_loss = 0.19
   * 1000 epochs
   * 21 layers
   * test results: true = 2615, false = 385
   * test accuracy = 87.2%
3. Sequence data 
   * original sequence length = 220, extending trained model for vectorized data
   * acc = 0.99, loss = 0.015, val_acc = 0.98, val_loss = 0.07
   * 100 epochs
   * 21 trained layers (vector -> class) +13 additional layers (seq -> vector)
   * test results: true = 2878, false = 122
   * test accuracy = 95.9%
