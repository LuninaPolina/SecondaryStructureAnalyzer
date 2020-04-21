Secondary structure prediction for sequences of fixed length = 90.

Data: train/valid/test = 40817/4535/11337 total samples.

Database: RNAcentral, prediction tool: CentroidFold.

Learning steps:

  * Step 1: parsed --> centroid
  * Step 2: predicted --> aligned
  * Step 3: parsed --> predicted --> aligned --> centroid
  
Metrics:
  * Precision = TW / (TW + FW)
  * Recall = TW / (TW + FB)
  * FMera = (2 * Precision * Recall) / (Precision + Recall)
  * LevDist -- levenshtein distance

Best models results:

| metrics 	| step1 | step2	| step3 |
|---------	|-----	|-----	|-----	|
| Precision | 85% 	| 85%  	| 95%  	|
| Recall  	| 89% 	| 90% 	| 83%  	|
| FMera    	| 86%  	| 87%  	| 88% 	|

LevDists for final model:

| LevDist | samples |
|---------|---------|
| 0       |   18%   |
| 1-5     |   22%   |
| 6-10    |   27%   |
| >10     |   25%   |
| error   |   8%    |
