# SOCCER: An Information-Sparse Discourse State Tracking Collection (NAACL 2021)
  *by Ruochen Zhang and Carsten Eickhoff*

This repository contains source code necessary to collect the dataset and reproduce the results presented in 
[SOCCER: An Information-Sparse Discourse State Tracking Collection in the Sports Commentary Domain](http://brown.edu/Research/AI/files/pubs/naacl21.pdf).

## Getting Started
### Requirements
Please install the prerequisites via

`pip install requirements.txt`

You will also need to [download the WebDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads) that matches your Chrome browser version.

### Collecting the Dataset
To collect the dataset

`python src/get_dataset.py -dp=<chromedriver path>`

The SOCCER dataset will be stored in the`data/` folder with train, validation and test splits.

### Running Baselines

#### GRU Classifier
To reproduce the team-level results of the GRU classifier 
```
cd model/GRU_classifier

./train_eval.sh
```

#### GPT-2 Variant
To reproduce the team-level and player-level results of the GPT-2 variant
```
cd model/GPT_variant

./train_eval.sh
``` 

## Citing
If you would like to cite this work, please refer to:
```bibtex
@inproceedings{zhang-eickhoff-2021-soccer,
    title = "{SOCCER}: An Information-Sparse Discourse State Tracking Collection in the Sports Commentary Domain",
    author = "Zhang, Ruochen  and
      Eickhoff, Carsten",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.342",
    pages = "4325--4333",
}
```