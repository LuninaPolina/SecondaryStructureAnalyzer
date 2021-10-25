# TRNA 6 classes classification

## Archaeal, bacterial, fungi, plant, animal and protist

 * Data: train/valid/test = 12000/1500/4500 total samples
 * Extention of the images-extended model

## Results on the test set
  
| res\lbl | Archea | Bacteria | Fungi | Plant | Animal | Protist |
|---------|--------|----------|-------|-------|--------|---------|
| Archea  | 739 	 | 5  	    | 2  	  | 0   	| 1   	 | 3   	   |
| B       | 21  	 | 711 	    | 0   	| 0  	  | 0   	 | 18  	   |
| F       | 10  	 | 0  	    | 680 	| 3  	  | 14   	 | 43  	   |
| Plant   | 4   	 | 6  	    | 3   	| 710 	| 7   	 | 20  	   |
| Animal  | 6  	   | 5  	    | 26 	  | 34  	| 626  	 | 53      |
| Protist | 26  	 | 19 	    | 8 	  | 26  	| 17  	 | 509 	   |

   * accuracy = 91.3%
   * precision<sub>Archea</sub> = 98.5%       recall<sub>Archea</sub> = 91.7%
   * precision<sub>Bacteria</sub> = 94.9%     recall<sub>Bacteria</sub> = 95.3%
   * precision<sub>Fungi</sub> = 90.7%        recall<sub>Fungi</sub> = 94.6%
   * precision<sub>Plant</sub> = 94.7%        recall<sub>Plant</sub> = 91.8%
   * precision<sub>Animal</sub> = 83.5%       recall<sub>Animal</sub> = 94.1%
   * precision<sub>Protist</sub> = 84.1%      recall<sub>Protist</sub> = 79.0%

Training time: 2400 sec
