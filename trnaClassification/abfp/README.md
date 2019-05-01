4 classes classification: archaeal, bacterial, fungi and plant tRNA sequences.

Data: train/valid/test = 8000/3000/1000 total samples

Approaches (on the same data):
  
**Images** 

Original sequences of different length. Parsing-provided images resized to 80x80

1. model_30_94
   * test results: true = 2799, false = 201
   * test accuracy = 93.3%
   
---------------------------------------------------------------------------------  

**Vectors**

Origial sequences of length 220. Parsing-provided uint32 vectors decompressed to bytes

1. model_51_88
   * test results: true = 2600, false = 400
   * test accuracy = 86.7%

--------------------------------------------------------------------------------- 

**Vectors extended to sequences**

Original sequences of length 220. Extending trained model for vectorized data (abfp/vectors/models/model_51_88)

1. model_18_96
   * test results: true = 2887, false = 113
   * test accuracy = 96.2%

--------------------------------------------------------------------------------- 

**Images extended to sequences**

Original sequences of length 220. Extending trained model for image data (abfp/images/models/model_30_94)

1. model_17_95
   * test results: true = 2872, false = 128
   * test accuracy = 95.7%

--------------------------------------------------------------------------------- 
--------------------------------------------------------------------------------- 

TODO

Models for sequence data were also evaluated on a big dataset with x total samples: 
link

**Vectors extended to sequences**

1. model_18_96
   * test results: 
   * test accuracy = 

--------------------------------------------------------------------------------- 

**Images extended to sequences**

1. model_17_95
   * test results:
   * test accuracy = 
