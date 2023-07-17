# Multi-aspect Enhanced Convolutional Neural Networks for Knowledge Graph Completion

This is the code of paper 
**Multi-aspect Enhanced Convolutional Neural Networks for Knowledge Graph Completion**. 

## Requirements
- python 3.8
- torch 1.9
- dgl 0.8

## Datasets
Available datasets are:
    
    FB15k-237
    WN18RR


## Reproduce the Results
To run a model execute the following command :
- FB15k-237
```python run.py --data FB15k-237 --th1 0.3 --th2 0.2 --conve_hid_drop 0.2 --feat_drop 0.2 --input_drop 0.2 --num_filt 200 --x_ops p.b.d --temperature 0.007```
- WN18RR
```python run.py --data wn18rr --th1 0.2 --th2 0.1 --conve_hid_drop 0.4 --feat_drop 0.1 --input_drop 0.2 --num_filt 250 --x_ops p.b.d --r_ops p.b.d --temperature 0.001```




## Acknowledgement
We refer to the code of [LTE](https://github.com/MIRALab-USTC/GCN4KGC) and [DGL](https://github.com/dmlc/dgl). Thanks for their contributions.
