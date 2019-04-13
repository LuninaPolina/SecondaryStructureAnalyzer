Binary classification: eukaryotic vs prokaryotic tRNA sequences. Sequence length = 220.

Vectorized data: train.csv, valid.csv, test.csv containing 35000, 15000 and 20000 samples respectively.

Models:
  
1. model_10_96
   * test results: TP = 9593, TN = 9485, FP = 515, FN = 407 
   * test accuracy = 94.4%
2. model_17_94
   * test results: TP = 8997, TN = 9730, FP = 270, FN = 1003 
   * test accuracy = 93.6%
3. model_083_97
   * test results: TP = 9715, TN = 9445, FP = 555, FN = 285 
   * test accuracy = 95.8%
4. model_084_9728
   * test results: TP = 9802, TN = 9470, FP = 530, FN = 198 
   * test accuracy = 96.4%
5. model_084_9728(tune)
   * test results: TP = 9815, TN = 9457, FP = 543, FN = 185 
   * test accuracy = 96.4%
6. model_091_96
   * test results: TP = 9671, TN = 9501, FP = 499, FN = 329 
   * test accuracy = 95.9%
7. model_096_96
   * test results: TP = 9605, TN = 9474, FP = 526, FN = 395 
   * test accuracy = 95.4%
