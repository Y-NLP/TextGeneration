# Enhancing MOBA Game Commentary Generation With Fine-Grained Prototype Retrieval

The codes of NLPCC2023 paper: "Enhancing MOBA Game Commentary Generation With Fine-Grained Prototype Retrieval". 
It can be found here: [paper](https://link.springer.com/chapter/10.1007/978-3-031-44693-1_65).

## Data Declaration
- **corpus folder:** retrieval corpus, consisting of train set
- **dataset folder**: train set，valid set，test set
- **key folder:** *digit.txt* is digit attribute; *name.txt* is name attribute

## Run the Code
Run the following code to construct the training dataset，and then use the BART to train according to the paper parameters：
```bash
python construct_training_data.py
```
## Citation
If you found this article helpful, please cite the paper.
````
@InProceedings{10.1007/978-3-031-44696-2_66,
author="Lai, Haosen
and Yu, Jiong
and Wang, Shuoxin
and Zhang, Dawei
and Wu, Sixing
and Zhou, Wei",
editor="Liu, Fei
and Duan, Nan
and Xu, Qingting
and Hong, Yu",
title="Enhancing MOBA Game Commentary Generation with Fine-Grained Prototype Retrieval",
booktitle="Natural Language Processing and Chinese Computing",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="850--862",
isbn="978-3-031-44696-2"
}
````