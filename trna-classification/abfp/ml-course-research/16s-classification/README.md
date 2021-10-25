# 16s rRNA 4 classes classification 

## Archaea, bacteriae, fungi and plantae

 * Data: train/test = 24519/8173 total samples
 * LSTM extending images-extended model

## Results on the test set

| res\lbl  | Archea | Plant | Fungi | Bacteria |
|----------|--------|-------|-------|----------|
| Archea   | 1782 	| 0  	  | 0  	  | 17  	   |
| Plant    | 21  	  | 1648 	| 192   | 88  	   |
| Fungi    | 13  	  | 166  	| 2001 	| 35  	   |
| Bacteria | 578  	| 12    | 2  	  | 1618 	   |


   * accuracy = 89.14%
   * mean precision: 86.97%
   * mean recall: 86.79%
   * mean f-score: 86.88%
   * precision<sub>Archea</sub> = 98.5%       recall<sub>Archea</sub> = 74.43%
   * precision<sub>Plant</sub> = 94.7%        recall<sub>Plant</sub> = 90.25%
   * precision<sub>Fungi</sub> = 90.7%        recall<sub>Fungi</sub> = 91.16%
   * precision<sub>Bacteria</sub> = 94.9%     recall<sub>Bacteria</sub> = 92.03%

Training time: 420 sec
