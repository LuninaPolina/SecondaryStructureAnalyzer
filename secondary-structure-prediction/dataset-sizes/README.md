# Secondary structure prediction

## Experiments with different train set sizes 

  * Neural network input data -- parsing matrices, reference data -- contact matrices; both transformed to black-and-white images 
  * 801 RNA sequences having length from 1 up to 100 from RNAstrand database
  * Dynamic train:test split (10%:90%, ... 90%:10%)
  * 4 Parallel ResNets each containing 5 ResUnits

  
## Results on the test sets


| train:test | Precision | Recall | F1  |
|------------|-----------|--------|-----|
| 10:90      | 58% 	     | 62%    | 59% |
| 20:80      | 61%  	   | 65%    | 62% |
| 30:70      | 64%  	   | 65%    | 63% |
| 40:60      | 65%       | 70%    | 66% |
| 50:50      | 65%       | 71%    | 67% |
| 60:40      | 68%       | 73%    | 69% |
| 70:30      | 71%       | 75%    | 72% |
| 80:20      | 69%       | 75%    | 71% |
| 90:10      | 71%       | 76%    | 72% |
