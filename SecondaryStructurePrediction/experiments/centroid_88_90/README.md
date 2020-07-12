Secondary structure prediction for sequences of length from 88 to 90.

Data: train/valid/test = 56143/6238/15595 total samples.

Database: RNAcentral, prediction tool: CentroidFold.

Learning steps:

  * Step 1: parsed --> centroid
  * Step 2: predicted --> aligned
  * Step 3: parsed --> predicted --> aligned --> centroid
  
Metrics:
  * Precision = TW / (TW + FW)
  * Recall = TW / (TW + FB)
  * FMera = (2 * Precision * Recall) / (Precision + Recall)

Results:

| metrics 	| original  | aligned   |
|-----------|-----------|-----------|
| Precision | 	 66%    |   81%  	|
| Recall  	|    78%    |   62% 	|
| FMera    	|    69%    |   68%  	| 

