This is the code for **GNN-RAG: Graph Neural Retrieval for Large Language Modeling Reasoning**.


![alt GNN-RAG: The GNN reasons over a dense subgraph to retrieve candidate answers, along
with the corresponding reasoning paths (shortest paths from question entities to answers). The
retrieved reasoning paths -optionally combined with retrieval augmentation (RA)- are verbalized
and given to the LLM for RAG](GNN-RAG.png "GNN-RAG")

The directory is the following:

|----`gnn` folder has the implementation of different KGQA GNNs. 

You can train your own GNNs or you can skip this folder and  use directly the GNN output (retrieved answer nodes) that we computed (`llm/results/gnn`).

|----`llm` folder has the implementation for RAG-based KGQA with LLMs. 

Please see details on how to reproduce results there. 

**Results**: We append all the results for Table 2: See `results/KGQA-GNN-RAG-RA` or `results/KGQA-GNN-RAG`. You can look at the actual LLM generations, as well as the KG information retrieved ("input" key) in predictions.jsonl.

## Get Started
We have simple requirements in `requirements.txt`. You can always check if you can run the code immediately.

The datasets as well as the pretrained LM (LMsr) are uploaded here: https://drive.google.com/drive/folders/1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp?usp=sharing

Please download them and extract them to the corresponding folders.

## Training
Please follow the guidelines and hyperparamters of the corresponding GNNs for training. See `scripts` on a training example.  

Otherwise, you can download released GNN models from here: https://drive.google.com/file/d/1p7eLSsSKkZQxB32mT5lMsthVP6R_3x1j/view
