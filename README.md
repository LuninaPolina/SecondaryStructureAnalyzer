For each experiment the folder structure should be approximately like this:

```
experimentName
|
└──data
|  └──datalinks.txt  
|      
├──models
|   |
|   └──modelName
|      ├──model.py
|      ├──model_test.py
|      ├──model_visualization.py
|      ├──weights.h5
|      └──trace.log
|      
└───scripts
|   ├──script1.py
|   └──...
|
└───README.md
```

Brief description:

* datalinks.txt -- file with links to some cloud storage with all neccessary data
* models folder contains all models for this data processing; naming notation is model_ab_cd, where ab -- 2 first numbers of val loss, cd -- 2 first numbers of val accuracy for this model
* scripts folder contains optional scripts for intermidiate data processing
* README.md file should provide all the evaluation results for each model


Links to articles with detailed explanation of the proposed approach motivations and usage:

* https://github.com/YaccConstructor/articles/blob/master/2019/bioinformatics/paper/Example.pdf
* https://github.com/YaccConstructor/articles/blob/master/2019/diploma/Polina%20Lunina/diploma.pdf
