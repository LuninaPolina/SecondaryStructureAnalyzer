# RNA secondary structure prediction

## Experiments on different datasets 

  * Neural network input data — parsing matrices, reference data — contact matrices; both transformed to black-and-white images
  * 10-fold cross-validation, all metrics are calculated for the best fold
  * Sequences lenghts 1-200
  * 3 models on 3 different datasets:
    * Main: data from RNAstrand database, 1091 samples, general model 
    * Pks: data from RNAstrand + Pseudobase databases, 1447 samples, model for predicting pseudoknots
    * Mps: data from PDB database, 712 samples, model for predicting multiplets
  * Postprocessing: binarization with threshold=0.6 and multiplets removal

## Results on the best valid sets

| Model | Best fold | Precision | Recall | F1  | Pknots | Pknots 0.25% | W-C pairs | G-U pairs | Wobble pairs |
|-------|-----------|-----------|--------|-----|--------|--------------|-----------|-----------|--------------|
| Main  | 3         | 75%       | 67%    | 70% | 0/11   | 3/11         | 1067/1650 | 68/185    | 53/135       |
| Pks   | 5         | 76%       | 66%    | 70% | 9/43   | 18/34        | 1628/2407 | 61/258    | 47/156       |
| Mps   |


## Global test results

F1 score on several popular databases

| Model | CRW | Spinzl | SRP | Rfam | PDB | Pseudobase | All data |
|-------|-----|--------|-----|------|-----|------------|----------|
| Main  | 69% | 73%    | 59% | 43%  | 74% | 42%        | 69%      |
| Pks   | 66% | 72%    | 56% | 48%  | 74% | 81%        | 68%      |
