Secondary structure prediction for sequences of length from 50 to 90.

Data: train/valid/test = 102122/11346/28367 total samples.

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
| Precision | 	 60%    |   71%  	|
| Recall  	|    72%    |   61% 	|
| FMera    	|    63%    |   63%  	| 

