# TRNA 9 types classification

## Actinobacteria, Bacteroidetes, Chlamidiae, Firmicutes, Fusobacteria, Proteobacteria, Spirochaetes, Tenericutes and other

  * Data: train/valid/test = 44000/5500/6750 total samples
  * Extension of the images-extended model
  * If the sequence type is not from these 8 ones than consider it as 'other' class
 
 ## Results on the test set
 
| res\lbl 	    | Actinobacteria | Bacteroidetes | Chlamidiae | Firmicutes | Fusobacteria | Proteobacteria | Spirochaetes | Tenericutes | other |
|---------------|----------------|---------------|------------|------------|--------------|----------------|--------------|-------------|-------|
|Actinobacteria | 562            | 4             | 6          | 21         | 0            | 13             | 4            | 0           | 140   |
|Bacteroidetes  | 15             | 467           | 3          | 9          | 3            | 35             | 11           | 6           | 201   |
|Chlamidiae     | 32             | 16            | 451        | 41         | 1            | 14             | 11           | 27          | 157   |
|Firmicutes     | 27             | 7             | 17         | 455        | 1            | 22             | 2            | 26          | 193   |
|Fusobacteria   | 10             | 7             | 7          | 35         | 527          | 21             | 20           | 7           | 116   |
|Proteobacteria | 29             | 15            | 3          | 30         | 1            | 509            | 2            | 1           | 160   |
|Spirochaetes   | 32             | 19            | 2          | 4          | 5            | 11             | 503          | 8           | 166   |
|Tenericutes    | 5              | 7             | 25         | 61         | 3            | 6              | 10           | 522         | 111   |
|other          | 52             | 33            | 2          | 30         | 1            | 32             | 12           | 2           | 586   |


   * accuracy = 67.9%
   * precision<sub>Actinobacteria</sub> = 73.6%       recall<sub>Actinobacteria</sub> = 74.9%
   * precision<sub>Bacteroidetes</sub> = 81.2%        recall<sub>Bacteroidetes</sub> = 62.3%
   * precision<sub>Chlamidiae</sub> = 87.4%      recall<sub>Chlamidiae</sub> = 60.1%
   * precision<sub>Firmicutes</sub> = 66.3%     recall<sub>Firmicutes</sub> = 60.7%
   * precision<sub>Fusobacteria</sub> = 97.2%      recall<sub>Fusobacteria</sub> = 70.3%
   * precision<sub>Proteobacteria</sub> = 76.8%       recall<sub>Proteobacteria</sub> = 67.9%
   * precision<sub>Spirochaetes</sub> = 87.5%        recall<sub>Spirochaetes</sub> = 67.1%
   * precision<sub>Tenericutes</sub> = 87.1%      recall<sub>Tenericutes</sub> = 69.6%
   * precision<sub>other</sub> = 32.0%      recall<sub>other</sub> = 78.1%

Training time: 13500 sec
