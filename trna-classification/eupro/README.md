# 2 classes tRNA classification

## Binary classification: eukaryotic VS prokaryotic sequences

 * Data: train/valid/test = 20000/5000/10000 total samples
 * 4 approaches on the same data
 * Fully connected dense neural networks with one convolutional layer for images processing

## Results on the test set
  
### Images 

Original sequences of different length. Parsing-provided images resized to 80x80


| res\lbl | P    | E    |
|---------|------|------|
| P       | 4925 | 304  |
| E       | 75   | 4696 |
  
   * accuracy = 96.2%
   * precision<sub>p</sub> = 94.2%        recall<sub>p</sub> = 98.5%
   * precision<sub>e</sub> = 98.4%        recall<sub>e</sub> = 93.9%

Training time: 2200 sec
   
---------------------------------------------------------------------------------  

### Vectors

Origial sequences of length 220. Parsing-provided uint32 vectors decompressed to bytes


| res\lbl | P    | E    |
|---------|------|------|
| P       | 4807 | 394  |
| E       | 193  | 4606 |
  
   * accuracy = 94.1%
   * precision<sub>p</sub> = 92.4%        recall<sub>p</sub> = 96.1%
   * precision<sub>e</sub> = 96.0%        recall<sub>e</sub> = 92.1%

Training time: 27000 sec

--------------------------------------------------------------------------------- 

### Vectors extended to sequences

Original sequences of length 220. Extending trained model for vectorized data (eupro/vectors/model)


| res\lbl | P    | E    |
|---------|------|------|
| P       | 4971 | 218  |
| E       | 29   | 4782 |
  
   * accuracy = 97.5%
   * precision<sub>p</sub> = 95.8%        recall<sub>p</sub> = 99.4% 
   * precision<sub>e</sub> = 99.4%        recall<sub>e</sub> = 95.6%

Training time: 1300 sec

--------------------------------------------------------------------------------- 

### Images extended to sequences

Original sequences of length 220. Extending trained model for image data (eupro/images/model)


| res\lbl | P    | E    |
|---------|------|------|
| P       | 4974 | 199  |
| E       | 26   | 4801 |
  
   * accuracy = 97.8%
   * precision<sub>p</sub> = 96.2%        recall<sub>p</sub> = 99.4%  
   * precision<sub>e</sub> = 99.4%        recall<sub>e</sub> = 99.5%

Training time: 3900 sec

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------

##Global test results

Models for sequence data were also evaluated on 2 complete tRNA databases (http://trna.ie.niigata-u.ac.jp/cgi-bin/trnadb/index.cgi and http://gtrnadb2009.ucsc.edu/).

Total samples: 615564 (prokaryotic: 527568, eukaryotic: 87996)

### Vectors extended to sequences

| res\lbl | P    	 | E     |
|---------|--------|-------|
| P       | 525464 | 3048  |
| E       | 2104   | 84948 |
  
   * accuracy = 99.2%
   * precision<sub>p</sub> = 99.4%        recall<sub>p</sub> = 99.6% 
   * precision<sub>e</sub> = 97.6%        recall<sub>e</sub> = 96.5%
   
---------------------------------------------------------------------------------

### Images extended to sequences

| res\lbl | P    	 | E     |
|---------|--------|-------|
| P       | 525426 | 2716  |
| E       | 2142   | 85280 |
  
   * accuracy = 99.2%
   * precision<sub>p</sub> = 99.5%        recall<sub>p</sub> = 99.6% 
   * precision<sub>e</sub> = 97.5%        recall<sub>e</sub> = 96.9%
