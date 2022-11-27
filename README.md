# 简介

This is a pytorch version of Long and Diverse Text Generation with Planning-based Hierarchical Variational Model, and it was rewrited based on [the tensorflow version projrct](https://github.com/ZhihongShao/Planning-based-Hierarchical-Variational-Model).

***<font color=Red size=4>Maybe there are some bug in this projrct, Hope you can find those mistake and solve them.</font>***

# Quick Start

*   Dataset
  
    Our dataset contains 119K pairs of product specifications and the corresponding advertising text. For more information, please refer to our paper.
    
*   Preprocess

    *   Download data from [here](https://drive.google.com/open?id=1vB0fT1ex2Tsid-i5s-jqdz9QUFbCh0CO) and unzip the file, which will create a new directory named `data`. The path to our dataset is `./data/data.jsonl`.
    *   We provided most preprocessed data under `./data/processed/` except pre-trained word embeddings which can be generated with the following command line:
    
    ```
    bash preprocess.sh
    ```

*   Train

    ```
    ./run.sh
    ```

*   Test

    ```
    ./test.sh
    ```


