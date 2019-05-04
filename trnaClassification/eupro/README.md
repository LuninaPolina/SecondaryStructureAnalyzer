Binary classification: eukaryotic vs prokaryotic tRNA sequences

Data: train/valid/test = 20000/5000/10000 total samples

Confusion matrix (CM) and metrics notations:

  * CM rows -- classification results, CM columns -- expecred labels for each class
  * accuracy = sum(CM<sub>pp</sub>) / sum(CM<sub>pq</sub>) for all p, q 
  * precision<sub>p</sub> = CM<sub>pp</sub> / sum(CM<sub>pq</sub>) for all q -- precision for fixed class p
  * recall<sub>p</sub> = CM<sub>pp</sub> / sum(CM<sub>qp</sub>) for all q -- recall for fixed class p

Approaches (on the same data):
  
**Images** 

Original sequences of different length. Parsing-provided images resized to 80x80

1. model_08_97

| res\lbl 	| P    	| E    	|
|---------	|------	|------	|
| P       	| 4925 	| 304  	|
| E       	| 75   	| 4696 	|
  
   * accuracy = 96.2%
   * precision<sub>p</sub> = 94.2%
   * precision<sub>e</sub> = 98.4%
   * recall<sub>p</sub> = 98.5%
   * recall<sub>e</sub> = 93.9%
   
---------------------------------------------------------------------------------  

**Vectors**

Origial sequences of length 220. Parsing-provided uint32 vectors decompressed to bytes

1. model_16_95

| res\lbl 	| P    	| E    	|
|---------	|------	|------	|
| P       	| 4807 	| 394  	|
| E       	| 193  	| 4606 	|
  
   * accuracy = 94.1%
   * precision<sub>p</sub> = 92.4%
   * precision<sub>e</sub> = 96.0%
   * recall<sub>p</sub> = 96.1%
   * recall<sub>e</sub> = 92.1%

--------------------------------------------------------------------------------- 

**Vectors extended to sequences**

Original sequences of length 220. Extending trained model for vectorized data (eupro/vectors/models/model_16_95)

1. model_05_98

| res\lbl 	| P    	| E    	|
|---------	|------	|------	|
| P       	| 4971 	| 218  	|
| E       	| 29   	| 4782 	|
  
   * accuracy = 97.5%
   * precision<sub>p</sub> = 95.8%
   * precision<sub>e</sub> = 99.4%
   * recall<sub>p</sub> = 99.4% 
   * recall<sub>e</sub> = 95.6%

--------------------------------------------------------------------------------- 

**Images extended to sequences**

Original sequences of length 220. Extending trained model for image data (eupro/images/models/model_08_97)

1. model_06_98

| res\lbl 	| P    	| E    	|
|---------	|------	|------	|
| P       	| 4974 	| 199  	|
| E       	| 26  	 | 4801 	|
  
   * accuracy = 97.8%
   * precision<sub>p</sub> = 96.2% 
   * precision<sub>e</sub> = 99.4%
   * recall<sub>p</sub> = 99.4% 
   * recall<sub>e</sub> = 99.5%
