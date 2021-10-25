# TRNA 8 types classification

## Actinobacteria, Bacteroidetes, Chlamidiae, Firmicutes, Fusobacteria, Proteobacteria, Spirochaetes, and Tenericutes

 * Data: train/valid/test = 32000/4000/6000 total samples
 * Extension of the images-extended model
 * If the sequence class is not from these 8 ones than remove it from the dataset
  
| res\lbl 	    | Actinobacteria | Bacteroidetes | Chlamidiae | Firmicutes | Fusobacteria | Proteobacteria | Spirochaetes | Tenericutes |
|---------------|----------------|---------------|------------|------------|--------------|----------------|--------------|-------------|
|Actinobacteria | 646            | 9             | 4          | 47         | 0            | 33             | 11           | 0           |
|Bacteroidetes  | 79             | 528           | 11         | 38         | 1            | 48             | 41           | 4           |
|Chlamidiae     | 87             | 29            | 472        | 57         | 2            | 37             | 27           | 39          |
|Firmicutes     | 66             | 15            | 15         | 563        | 2            | 45             | 20           | 24          |
|Fusobacteria   | 31             | 18            | 4          | 76         | 517          | 43             | 39           | 22          |
|Proteobacteria | 48             | 24            | 6          | 78         | 1            | 581            | 8            | 4           |
|Spirochaetes   | 85             | 23            | 12         | 30         | 3            | 16             | 552          | 29          |
|Tenericutes    | 17             | 10            | 40         | 98         | 4            | 27             | 42           | 512         |

   * accuracy = 73.3%
   * precision<sub>Actinobacteria</sub> = 61.0%       recall<sub>Actinobacteria</sub> = 86.1%
   * precision<sub>Bacteroidetes</sub> = 80.5%        recall<sub>Bacteroidetes</sub> = 70.4%
   * precision<sub>Chlamidiae</sub> = 83.7%      recall<sub>Chlamidiae</sub> = 62.9%
   * precision<sub>Firmicutes</sub> = 57.0%     recall<sub>Firmicutes</sub> = 75.1%
   * precision<sub>Fusobacteria</sub> = 97.5%      recall<sub>Fusobacteria</sub> = 68.9%
   * precision<sub>Proteobacteria</sub> = 70.0%       recall<sub>Proteobacteria</sub> = 77.5%
   * precision<sub>Spirochaetes</sub> = 74.6%        recall<sub>Spirochaetes</sub> = 73.6%
   * precision<sub>Tenericutes</sub> = 80.8%      recall<sub>Tenericutes</sub> = 68.3%

Training time: 9000 sec
