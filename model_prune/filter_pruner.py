import torch
from torch.autograd import Variable
import torch.optim as optim
from operator import itemgetter
from heapq import nsmallest
from prune_utils import *

def total_num_filters(model):
    filters = 0
    for name1, module1 in model._modules.items():
        if isinstance(module1, torch.nn.modules.conv.Conv2d):
            filters = filters + module1.out_channels
        for name2, module2 in module1._modules.items():
            if isinstance(module2, torch.nn.modules.conv.Conv2d):
                filters = filters + module2.out_channels
            for name3, module3 in module2._modules.items():
                if isinstance(module3, torch.nn.modules.conv.Conv2d):
                    filters = filters + module3.out_channels
                for name4, module4 in module3._modules.items():
                    if isinstance(module4, torch.nn.modules.conv.Conv2d):
                        filters = filters + module4.out_channels
    return filters


class FilterPruner:
    def __init__(self, model, train_dataloader, test_dataloader, criterion, useCuda):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.useCuda = useCuda
        self.reset()

    def reset(self):
        self.filter_ranks = {}
        self.index_to_layername = {}
        self.layername_to_index = {}
        self.hooks = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def hook_generator(self, layername):
        def hook_func(grad):
            activation_index = self.layername_to_index[layername]
            # print layername,'hook'
            activation = self.activations[activation_index]
            # print layername,grad.size(),activation_index,self.activations[activation_index].size()
            values = \
                torch.sum((activation * grad), dim=0, keepdim=True). \
                    sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
            # Normalize the rank by the filter dimensions
            values = \
                values / (activation.size(0) * activation.size(2) * activation.size(3))

            if activation_index not in self.filter_ranks:
                if self.useCuda:
                    self.filter_ranks[activation_index] = \
                        torch.FloatTensor(activation.size(1)).zero_().cuda()
                else:
                    self.filter_ranks[activation_index] = \
                        torch.FloatTensor(activation.size(1)).zero_()

            self.filter_ranks[activation_index] += values

        return hook_func

    def attach_action_and_hook(self, activation, activation_index, layer_name):
        hook = self.hook_generator(layer_name)
        h = activation.register_hook(hook)
        self.hooks.append(h)
        self.activations.append(activation)
        # binding layer name and index
        if not activation_index in self.index_to_layername.keys():
            self.index_to_layername[activation_index] = layer_name
            self.layername_to_index[layer_name] = activation_index

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)  # [(actv_idx, channel_idx, value), ...]

        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        return filters_to_prune_per_layer

    def forward(self,x):
        '''
        :param x: a batch of input images, Tensor of shape (N,C,H,W), as default of PyTorch
        :return: model's output
        '''
        raise NotImplementedError

    def train(self,epochs):
        # optimizer should be updated since model's parameters changes after pruning
        optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        for i in range(epochs):
            print "Epoch: ", i
            self.model.train()
            for j, (inputs, labels) in enumerate(self.train_dataloader):
                self.model.zero_grad()
                if self.useCuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                loss = self.criterion(self.model(inputs), labels)
                loss.backward()
                optimizer.step()
            self.model.eval()
            self.test()
        print "Finished fine tuning."

    def test(self):

        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(self.test_dataloader):
            if self.useCuda:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            output = self.model(inputs)
            pred = output.data.max(1)[1]

            correct += pred.eq(labels.data).sum()
            total += labels.size(0)

        print "Accuracy :", float(correct) / total

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.reset()

        for inputs, labels in self.train_dataloader:
            self.model.zero_grad()
            if self.useCuda:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            output = self.forward(inputs)  # compute filter activation, register hook function
            self.criterion(output, labels).backward()  # compute filter grad by hook, and grad*activation

        self.remove_hooks()
        self.normalize_ranks_per_layer()
        return self.get_prunning_plan(num_filters_to_prune)

    def prune_conv_layer(self,model, prune_target):
        '''
        :param model: model to  prune
        :param prune_target: dictionary of "layer name tuple : list of filters to prune in the layer"
        :return: pruned model
        '''
        raise NotImplementedError


    def perform_prune(self, save_dir, proportion=0.5, num_filters_per_iter=256, epochs_after_perprune=10,
                      epochs_after_wholeprune=10):
        '''
        :param save_dir: folder to save pruned model
        :param proportion: proportion of remained model's filters, float
        :param num_filters_per_iter: num of filters to prune per iter, int
        :param epochs_after_perprune: num of finetune epochs after per prune, int
        :param epochs_after_wholeprune: num of finetune epochs after all prune, int
        '''
        # create a temp folder in save_dir
        if not os.path.exists(os.path.join(save_dir,'temp')):
            os.makedirs(os.path.join(save_dir,'temp'))

        # Get the accuracy before prunning
        if self.useCuda:
            print 'USING CUDA...'
            self.model  = self.model.cuda()
        self.model.eval()
        self.test()

        number_of_filters = total_num_filters(self.model)
        iterations = int(float(number_of_filters) / num_filters_per_iter)
        iterations = int(iterations * (1-proportion))

        print 'total num of filters:', number_of_filters
        print "Number of prunning iterations to reduce %s filters: %s"%(proportion, iterations)

        for _ in range(iterations):

            self.model.eval()
            prune_targets = self.get_candidates_to_prune(num_filters_per_iter)
            prune_targets_by_layername = {self.index_to_layername[idx]:prune_targets[idx] for idx in prune_targets.keys()}

            # # uncomment to see details of pruning targets
            # for layer_name in prune_targets_by_layername.keys():
            #     print layer_name, ':' ,prune_targets_by_layername[layer_name], ','

            layers_prunned = {self.index_to_layername[idx]: len(prune_targets[idx]) for idx in prune_targets.keys()}
            #
            print "Layers that will be pruned", layers_prunned
            print "Pruning filters.. "

            model = self.model.cpu()
            model = self.prune_conv_layer(model, prune_targets_by_layername)
            torch.save(model, os.path.join(save_dir,'temp','model_pruned_test.pth'))
            model = torch.load(os.path.join(save_dir,'temp','model_pruned_test.pth'))

            self.model = model

            message = str(100 * float(total_num_filters(self.model)) / number_of_filters) + "%"
            print "Filters remain", str(message)

            # to use multi-GPU when fintuning and tesing
            if self.useCuda:
                self.model = torch.nn.DataParallel(self.model).cuda()

            self.model.eval()
            self.test()

            print "Fine tuning to recover from pruning iteration."
            self.model.train()

            self.train(epochs=epochs_after_perprune)
            if self.useCuda:
                self.model = self.model.module  # stop DataParallel


        print "Finished. Going to fine tune the model a bit more"
        if self.useCuda:
            self.model.cpu()
            self.model = torch.nn.DataParallel(self.model).cuda()
        self.train(epochs=epochs_after_wholeprune)
        torch.save(self.model.cpu(), os.path.join(save_dir,'model_pruned_test.pth'))
