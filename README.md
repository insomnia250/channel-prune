### This is a PyTorch implementation of [[1611.06440 Pruning Convolutional Neural Networks for Resource Efficient Inference]](https://arxiv.org/abs/1611.06440).

#### 3x model size reducing can be achieved in the [Kaggle's cat-vs-dog dataset](https://www.kaggle.com/c/dogs-vs-cats/data) with rather tiny accuracy loss.

#### This project was modified from [https://github.com/jacobgil/pytorch-pruning](https://github.com/jacobgil/pytorch-pruning) and some main differences are:
- Pruning is done in one single pass rather than sequentially.
- FilterPruner and PruningFineTuner in the referred original project are merged to FilterPruner class for clarity and simplicity.
- **InceptionV3**, **Inception_Resnet_V1**, **Resnet50** are pruned as examples and you can define a new FilterPruner for your custom model.

### Note:
- You can use PyTorch's pretrained **Resnet50** or **InceptionV3** as a base model and prune them on the cat-vs-dog dataset mentioned before.\
(See prune_InceptionV3_example.py and prune_Resnet50_example.py)
- To prune a new model, you need to define a **forward** function and a **prune_conv_layer** function under **FilterPruner** according to your model's architecture. 