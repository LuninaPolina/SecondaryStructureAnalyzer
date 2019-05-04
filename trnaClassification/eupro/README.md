Binary classification: eukaryotic vs prokaryotic tRNA sequences

Data: train/valid/test = 20000/5000/10000 total samples

Approaches (on the same data):
  
**Images** 

Original sequences of different length. Parsing-provided images resized to 80x80

1. model_08_97
   * test results: TP = 4925, TE = 4696, FP = 304, FE = 75 
   * test accuracy = 96.2%
   
---------------------------------------------------------------------------------  

**Vectors**

Origial sequences of length 220. Parsing-provided uint32 vectors decompressed to bytes

1. model_16_95
   * test results: TP = 4807, TE = 4606, FP = 394, FE = 193 
   * test accuracy = 94.1%

--------------------------------------------------------------------------------- 

**Vectors extended to sequences**

Original sequences of length 220. Extending trained model for vectorized data (eupro/vectors/models/model_16_95)

1. model_05_98
   * test results: TP = 4971, TE = 4782, FP = 218, FE = 29 
   * test accuracy = 97.5%

--------------------------------------------------------------------------------- 

**Images extended to sequences**

Original sequences of length 220. Extending trained model for image data (eupro/images/models/model_08_97)

1. model_062_987
   * test results: TP = 4974, TE = 4801, FP = 199, FE = 26 
   * test accuracy = 97.8%
2. model_065_988
   * test results: TP = 4966, TE = 4780, FP = 220, FE = 34 
   * test accuracy = 97.5%

--------------------------------------------------------------------------------- 
--------------------------------------------------------------------------------- 

Models for sequence data were also evaluated on 2 full tRNA databases.

Database1: http://gtrnadb2009.ucsc.edu/ ; eukaryotic:procaryotic = 84464:26645
Database2: http://trna.ie.niigata-u.ac.jp/ ; eukaryotic:procaryotic = 3534:507388

Test results:

|                 	|     	| TP     	| TE    	| FP   	| FE   	| Accuracy 	|
|------------------	|------	|----------	|----------	|------	|------	|----------	|
| Images          	| db1 	| 26588  	| 82782 	| 1682 	| 57   	| 98.4%    	|
| (model_065_988) 	| db2 	| 504395 	| 2137  	| 1397 	| 2993 	| 99.1%    	|
|------------------	|------	|----------	|----------	|------	|------	|----------	|
| Images          	| db1 	| 26558  	| 83058 	| 1406 	| 87   	| 98.7%    	|
| (model_062_987) 	| db2 	| 505089 	| 2225  	| 1309 	| 2299 	| 99.3%    	|
|------------------	|------	|----------	|----------	|------	|------	|----------	|
| Vectors         	| db1 	| 26580  	| 82815 	| 1649 	| 65   	| 98.5%    	|
| (model_05_98)   	| db2 	| 505130 	| 2139  	| 1395 	| 2258 	| 99.3%    	|
|------------------	|------	|----------	|----------	|------	|------	|----------	|


