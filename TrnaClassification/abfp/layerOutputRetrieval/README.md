### Retrieval of intermediate layer results

intermediate_layers_outputs.py is intended for retrieving results of 
border layers of the models presented in imagesExtended and vectorsExtended folders.
The script addresses this [issue](https://github.com/LuninaPolina/SecondaryStructureAnalyzer/issues/5).

Main functions of intermediate_layers_outputs.py:

* `visualize_vec_model()` - retrieves intermediate layer results of 
the model presented in vectorsExtended, parses them into images 
and saves them into a specified directory. Examples of this function's 
work are presented in intermediate_output_vec.zip. 
You have to pass a sequence to this function. For sequence examples see 
`datalinks.txt` in vectorsExtended.

* `save_img_model_results()` - retrieves intermediate layer results of 
the model presented in imagesExtended, and saves them as numpy arrays 
into a specified directory. You have to pass a sequence to this function. 
For sequence examples see 
`datalinks.txt` in imagesExtended.

* `load_img_model_results()` - loads the results of work of the previous method and 
returns them as a list of numpy arrays.
