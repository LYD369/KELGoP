# KELGoP

This is the code and data for **A Framework of Knowledge Graph-Enhanced Large Language Model Based on Global Planning**.

## Setup

1. Install the required libraries following [`requirements.txt`](https://github.com/LYD369/KELGoP/blob/main/requirements.txt).

2. Follow the steps in [ToG/Freebase](https://github.com/GasolSun36/ToG/tree/main/Freebase) to setup Freebase.
   
3. Follow the steps in [`model/BLINK/README.md`](https://github.com/LYD369/KELGoP/blob/main/model/BLINK/README.md) and [`model/GNN_RAG/README.md`](https://github.com/LYD369/KELGoP/blob/main/model/GNN_RAG/README.md) to setup the required models.

## Run

WebQSP:

```console
# Triple/quadruple-based KG searching
CUDA_VISIBLE_DEVICES=0 python KELGoP.py --dataset webqsp --KG_retriever 2hoptriple --reference_num 10
# Relation-based KG searching
CUDA_VISIBLE_DEVICES=0 python KELGoP.py --dataset webqsp --KG_retriever 2hoptriple --reference_num 13
# GNN-based KG searching
CUDA_VISIBLE_DEVICES=0 python KELGoP.py --dataset webqsp --KG_retriever gnn
```

CWQ:

```console
# Triple/quadruple-based KG searching
CUDA_VISIBLE_DEVICES=0 python KELGoP.py --dataset cwq --KG_retriever 2hoptriple --reference_num 10
# Relation-based KG searching
CUDA_VISIBLE_DEVICES=0 python KELGoP.py --dataset cwq --KG_retriever 2hoptriple --reference_num 13
# GNN-based KG searching
CUDA_VISIBLE_DEVICES=0 python KELGoP.py --dataset cwq --KG_retriever gnn
```

## Cite

```
@ARTICLE{11275658,
    author={Li, Yading and Song, Dandan and Tian, Yuhang and Wang, Hao and Zhou, Changzhi and Zhang, Shuhao},
    journal={IEEE Transactions on Knowledge and Data Engineering}, 
    title={A Framework of Knowledge Graph-Enhanced Large Language Model Based on Global Planning}, 
    year={2026},
    volume={38},
    number={2},
    pages={736-748},
    doi={10.1109/TKDE.2025.3639599}}
```
