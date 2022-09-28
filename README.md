## NLGNN_PyG

Reimplementation of TPAMI 2022 paper "[Non-Local Graph Neural Networks](https://arxiv.org/abs/2005.14612l)" based on PyTorch and PyTorch Geometric (PyG).



## Run

```
python main.py
```



## Note

- Currently, the number of local embedding layers and Conv1d layers are both set as 2 referring to the paper.

- We can change the local embedding mode among 'mlp', 'gcn', and 'gat' in main.py.

  

