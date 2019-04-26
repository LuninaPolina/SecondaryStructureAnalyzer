Binary classification: eukaryotic vs prokaryotic tRNA sequences

Data: train/valid/test = 20000/5000/10000 total samples

Approaches (on the same data):
  
Images 
original sequences of different length, parsing-provided images were resized to 80x80
1. model_08_97
   * test results: TP = 4925, TN = 4696, FP = 304, FN = 75 
   * test accuracy = 96.2%

Vectors
origial sequences length = 220, parsing-provided uint32 vectors to bytes
1. model_16_95
   * test results: TP = 4807, TN = 4606, FP = 394, FN = 193 
   * test accuracy = 94.1%

Vectors extended to sequences
original sequence length = 220
extending trained model for vectorized data (eupro/vectors220/models/model_16_95)
1. model_05_98
   * test results: TP = 4971, TN = 4782, FP = 218, FN = 29 
   * test accuracy = 97.5%

Images extended to sequences
original sequence length = 220
extending trained model for image data (eupro/images80/models/model_08_97)

1. model_062_987
   * test results: TP = 4974, TN = 4801, FP = 199, FN = 26 
   * test accuracy = 97.8%
2. model_065_988
   * test results: TP = 4966, TN = 4780, FP = 220, FN = 34 
   * test accuracy = 97.5%

Models for sequence data were also evaluated on a big dataset with 65000 total samples: 
https://mega.nz/#!pbo3RSTD!Ag0qei1yHZ39CmYOQSuu0mqcJazA5E9mNscTIuimmNQ

Vectors extended to sequences

1. model_05_98
   * test results: TP = 32399, TN = 31738, FP = 759, FN = 104
   * test accuracy = 98.7%

Images extended to sequences

1. model_062_987
   * test results: TP = 32399, TN = 31839, FP = 658, FN = 104
   * test accuracy = 98.8%
2. model_065_988
   * test results: TP = 32397, TN = 31752, FP = 745, FN = 106
   * test accuracy = 98.7%

