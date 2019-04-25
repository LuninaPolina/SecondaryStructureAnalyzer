Binary classification: eukaryotic vs prokaryotic tRNA sequences

Data: train/valid/test = 20000/5000/10000 total samples

Layers count includes normalization and activation, without input layer

Approaches (on the same data):
  
1. Image data 
   * original sequences of different length, parsing-provided images were resized to 80x80
   * acc = 0.99, loss = 0.008, val_acc = 0.97, val_loss = 0.08
   * 200 epochs
   * 20 layers
   * test results: TP = 4925, TN = 4696, FP = 304, FN = 75
   * test accuracy = 96.2%
2. Vectorized data
   * origial sequences length = 220, parsing-provided uint32 vectors to bytes
   * acc = 0.98, loss = 0.02, val_acc = 0.95, val_loss = 0.16
   * 1500 epochs
   * 21 layers
   * test results: TP = 4807, TN = 4606, FP = 394, FN = 193
   * test accuracy = 94.1%
3. Sequence data 
   * original sequence length = 220, extending trained model for vectorized data
   * acc = 0.99, loss = 0.006, val_acc = 0.98, val_loss = 0.05
   * 100 epochs
   * 21 trained layers (vector -> class) + 13 additional layers (seq -> vector)
   * test results: TP = 4971, TN = 4782, FP = 218, FN = 29
   * test accuracy = 97.5%
