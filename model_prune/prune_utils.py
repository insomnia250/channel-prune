import os
import numpy as np

def prune_conv_weight(module, prepruned_index, prune_index):
    ori_weight = module.weight.data  # shape of (out_c, in_c ,kernel_1, kernel_2)

    # prune due to previous layer
    remained_in_index = [x for x in range(ori_weight.size(1)) if x not in prepruned_index]

    # prune due to current layer
    remained_out_index = [x for x in range(ori_weight.size(0)) if x not in prune_index]

    module.in_channels = module.in_channels - len(prepruned_index)
    module.out_channels = module.out_channels - len(prune_index)

    remained_weight = ori_weight[remained_out_index, :, :, :]
    remained_weight = remained_weight[:, remained_in_index, :, :]
    module.weight.data = remained_weight

    if not module.bias is None:
        ori_bias = module.bias.data
        remained_bias = ori_bias[remained_out_index,]
        module.bias.data = remained_bias
    return module

def prune_bn_weight(module, prepruned_index):
    ori_weight = module.weight.data  # shape of (num_features, )
    ori_bias = module.bias.data  # shape of (num_features, )
    ori_running_mean = module.running_mean  # shape of (num_features, ) bn.running_mean is a Tensor, not Variable
    ori_running_var = module.running_var  # shape of (num_features, )  a Tensor, not Variable

    # prune due to previous layer
    remained_index = [x for x in range(ori_weight.size(0)) if x not in prepruned_index]

    module.num_features = module.num_features - len(prepruned_index)

    module.weight.data = ori_weight[remained_index,]
    module.bias.data = ori_bias[remained_index,]
    module.running_mean = ori_running_mean[remained_index,]
    module.running_var = ori_running_var[remained_index,]
    return module

def prune_fc_weight(module, prepruned_index):
    ori_weight = module.weight.data  # shape of (num_classes, num_features)

    # prune due to previous layer
    remained_index = [x for x in range(ori_weight.size(1)) if x not in prepruned_index]

    module.in_features = module.in_features - len(prepruned_index)

    module.weight.data = ori_weight[:,remained_index]
    return module

def concat_index(ori_out_channels, prune_indexs):
    '''
    :param ori_out_channels: list [32,32,64,...]
    :param prune_indexs: list of prune_index in previous layers(these layers are to be concat) [[1,4], [], [2], ...]
    :return: prune_index in concat layer  [1,4,30]
    '''
    channels_cated = []
    for i,idxs in enumerate(prune_indexs):
        if i==0:
            channels_cated += (prune_indexs[i])
        else:
            pre_channels = sum(ori_out_channels[:i])
            channels_cated += ([x+pre_channels for x in prune_indexs[i]])
    return channels_cated

