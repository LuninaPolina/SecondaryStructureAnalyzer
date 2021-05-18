# Secondary structure analysis research: formal grammars + neural networks


## Solution structure


### Parsing tool

* [Compiled version](https://mega.nz/file/dSw0hB6I#deVhTgQDbV6LAPS8tRKZr_2L6az3i8fd8ey_IextiYo)
* [Source code](https://github.com/YaccConstructor/YaccConstructor/tree/Rider)


#### Requirements (Ubuntu)

* Install mono
* Install .net framework 4.7.2 via wine
* Install dotnet-sdk-3.0
* `dotnet add package FSharp.Core --version 4.5.0`


#### Run

`./YaccConstructor/src/SecondaryStructureExtracter/bin/Debug/SecondaryStructureExtracter.exe -g 'grammar.txt' -i 'input_sequences.fasta' -o 'output_dir/'`

Additional argument: output file format -f csv (default -- bmp)

Grammar example is presented in the repo root folder


### Neural networks


#### Requirements

* Python 3
* TensorFlow-gpu 1x
* Keras, scikit-image


## Repository description


### Experiments

* **16S detection** -- binary classifier that separates true 16s RNA sequences from random parts of genome
* **Chimeras detection** -- not comleted research for chimeric sequences search in biological databases
* **Secondary structure prediction** -- model for predicting RNA secondary structure contact map from parsing provided matrix
* **TRNA classification** -- solutions for several trna classification tasks with small amount of classes


### Content

* Links to all required datasets and trained models weights are presented in data/datalinks.txt files 
* Models code and logs are presented in models/ folder
* All useful data processing scripts are stored in scripts/ folder


## Papers

* Grigorev S., Lunina P. The Composition of Dense Neural Networks and Formal Grammars for Secondary Structure Analysis //BIOINFORMATICS. – 2019. – С. 234-241.
* Grigorev S., Lunina P. Improved Architecture of Artificial Neural Network for Secondary Structure Analysis //BMC Bioinformatics. – 2019. – Т. 20. – №. S17. – С. P2.
* Lunina P., Grigorev S. On Secondary Structure Analysis by Using Formal Grammars and Artificial Neural Networks //International Meeting on Computational Intelligence Methods for Bioinformatics and Biostatistics. – Springer, Cham, 2019. – С. 193-203.
* Grigorev S., Kutlenkov D., Lunina P. Secondary structure prediction by combination of formal grammars and neural networks //BMC Bioinformatics. – 2020. – Т. 21. – №. SUPPL 20.
* [Комбинирование нейронных сетей и синтаксического анализа для предсказания вторичных структур генетических цепочек](https://github.com/YaccConstructor/articles/blob/master/2020/diploma/LuninaPolina/text/diploma.pdf)
* [Комбинирование нейронных сетей и синтаксического анализа для обработки вторичной структуры последовательностей](https://github.com/YaccConstructor/articles/blob/master/2019/diploma/Polina%20Lunina/diploma.pdf)
* [Комбинирование нейронных сетей и синтаксического анализа для предсказания вторичной структуры РНК](https://github.com/YaccConstructor/articles/tree/master/2021/diploma/Polina%20Lunina)


-----
Contact: lunina_polina@mail.ru
