# Secondary structure prediction: experiments on different datasets 


## Description

  * Neural network input data -- parsing matrices, reference data -- contact matrices; both transformed to black-and-white images
  * 10-fold cross-validation, all metrics are calculated for the best validation set
  * Sequences lenghts 1-200
  * 3 models on 3 different datasets:
    * Main: data from RNAstrand database, 1091 samples, general model 
    * Pks: data from RNAstrand + Pseudobase databases, 1447 samples, model for predicting pseudoknots
    * Mps: data from PDB database, 712 samples, model for predicting multiplets

## Results

| Model | Precision | Recall | F1  | Pknots | Pknots 0.25% | W-C pairs | G-U pairs | Wobble pairs |
|-------|-----------|--------|-----|--------|--------------|-----------|-----------|--------------|
| Main  | 75%       | 67%    | 70% | 0/11   | 3/11         | 1067/1650 | 68/185    | 53/135       |
| Pks   |
| Mps   |