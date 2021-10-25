# Secondary structure conatcts prediction


## Data

  * 801 RNA sequences having length from 1 up to 100 from RNAstrand database
  * Dynamic train:test split (10%:90%, ... 90%:10%)
  * Neural network input data -- parsing matrices, reference data -- contact matrices; both transformed to black-and-white images


## Neural Network

  * 4 identical parallel ResNets united by weighted sum of their outputs and 1 final ResUnit
  * Each ResNet consists of 5 ResUnits with 'add' shortcut connection
  * Each ResUnit contains 5 convolutional layers with kernels (13, 11, 9, 7, 5) and filters (12, 10, 8, 6, 1) respectively
  * Dropout, L2 regularization, Adagrad optimizer, loss based on f1 minimization 

  
## Results 


| train:test | Precision  | Recall     | F1         |
|------------|------------|------------|------------|
| 10:90      | 57.7% 	   | 62.1%      | 58.5%      |
| 20:80      | 61.3%  	   | 65.0%      | 62.0%      |
| 30:70      | 63.5%  	   | 65.2%      | 63.1%      |
| 40:60      | 64.9%      | 70.2%      | 66.1%      |
| 50:50      | 64.9%      | 70.8%      | 66.7%      |
| 60:40      | 67.9%      | 72.5%      | 69.3%      |
| 70:30      | 71.2%      | 74.6%      | 71.9%      |
| 80:20      | 68.8%      | 75.0%      | 70.8%      |
| 90:10      | 71.0%      | 75.6%      | 72.4%      |
