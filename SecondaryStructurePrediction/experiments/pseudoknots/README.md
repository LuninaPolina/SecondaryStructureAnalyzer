Secondary structure prediction for sequences of length from 50 to 90 with pseudoknots.

Data: train/valid/test = 195/21/50 total samples.

Databases: RNAcentral, Pseudobase.
  
Metrics:
  * Precision = TW / (TW + FW)
  * Recall = TW / (TW + FB)
  * FMera = (2 * Precision * Recall) / (Precision + Recall)

Results:

| metrics 	| original  |
|-----------|-----------|
| Precision | 	 74%    |
| Recall  	|    73%    |
| FMera    	|    71%    | 

