6 classes classification: archaeal, bacterial, fungi, plant, animal and protist tRNA sequences.

Data: train/valid/test = 12000/1500/4500 total samples

Confusion matrix (CM) and metrics notations:

  * CM rows -- classification results, CM columns -- expecred labels for each class
  * accuracy = sum(CM<sub>pp</sub>) / sum(CM<sub>pq</sub>) for all p, q 
  * precision<sub>p</sub> = CM<sub>pp</sub> / sum(CM<sub>pq</sub>) for all q -- precision for fixed class p
  * recall<sub>p</sub> = CM<sub>pp</sub> / sum(CM<sub>qp</sub>) for all q -- recall for fixed class p
  
| res\lbl 	| Archea | Bacteria | Fungi | Plant | Animal | Protist |
|---------	|-----	 |-----	    |-----	|-----	|-----	 |-----	   |
| Archea   	| 739 	 | 5  	    | 2  	| 0   	| 1   	 | 3   	   |
| B       	| 21  	 | 711 	    | 0   	| 0  	| 0   	 | 18  	   |
| F       	| 10  	 | 0  	    | 680 	| 3  	| 14   	 | 43  	   |
| Plant    	| 4   	 | 6  	    | 3  	| 710 	| 7   	 | 20  	   |
| Animal    | 6  	 | 5  	    | 26 	| 34  	| 626  	 | 53      |
| Protist   | 26  	 | 19 	    | 8 	| 26  	| 17  	 | 509 	   |

   * accuracy = 91.3%
   * precision<sub>Archea</sub> = 98.5%       recall<sub>Archea</sub> = 91.7%
   * precision<sub>Bacteria</sub> = 94.9%     recall<sub>Bacteria</sub> = 95.3%
   * precision<sub>Fungi</sub> = 90.7%        recall<sub>Fungi</sub> = 94.6%
   * precision<sub>Plant</sub> = 94.7%        recall<sub>Plant</sub> = 91.8%
   * precision<sub>Animal</sub> = 83.5%       recall<sub>Animal</sub> = 94.1%
   * precision<sub>Protist</sub> = 84.1%      recall<sub>Protist</sub> = 79.0%

Training time: 2400 sec