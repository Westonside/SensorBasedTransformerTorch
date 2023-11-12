import math

import numpy as np
import torch
from torch import nn, Tensor


class TransformerModel(nn.Module):

    def __init__(self,input_shape, activityCount, projection_dim = 192,patchSize = 16,timeStep = 16,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31], mlp_head_units = [1024],dropout_rate = 0.3,useTokens = False):
        super().__init__()
        self.model_type = 'Transformer'

        self.projectionHalf = projection_dim // 2
        self.projectionQuarter = projection_dim // 4
        self.dropPathRate = np.linspace(0, dropout_rate * 10, len(convKernels)) * 0.1
        self.transformer_units = [
            projection_dim * 2,
            projection_dim, ]
        #make an input layer that takes in the input and projects it to the correct size
        self.flatten = nn.Flatten() # flatten
        self.layer_1 = nn.linear(input_shape[0] * input_shape[1],projection_dim)
        self.input_layer = nn.Linear()
        self.num_patches = self.patches.shape[1]
        self.transform_layers = nn.ModuleDict() # add items as you go
        self.dropOut = nn.Dropout(dropout_rate)
        self.mlp_head_units = mlp_head_units

        for layerIndex, kernelLength in enumerate(convKernels):
            """
                This applies layer normalization to the input data to reduce the range
                Layer normalization works by:
                1. Take input array
                2. Subtract the mean of the input array from each value in the array
                3. Divide each value in the array by the standard deviation of the array
                This is done column wise for all items in the batch
            
            """
            first_layer_name = f"layer_{layerIndex}_norm"
            x1 = nn.LayerNorm(eps=1e-6, normalized_shape=projection_dim) # the get the dimensions of the input except the batch size
            self.transform_layers[first_layer_name] = x1
            # next is the multihead attention

            acc_attention_name = f"layer_{layerIndex}_acc_attention"
            acc_attention = SensorMultiHeadAttention(self.projectionQuarter, num_heads, 0, self.projectionHalf, dropout_rate, self.dropPathRate[layerIndex])
            self.transform_layers[acc_attention_name] = acc_attention

            gyro_attention_name = f"layer_{layerIndex}_gyro_attention"
            gyro_attention = SensorMultiHeadAttention(self.projectionQuarter, num_heads, self.projectionHalf, projection_dim, dropout_rate, self.dropPathRate[layerIndex])
            self.transform_layers[gyro_attention_name] = gyro_attention

            # the next normalization layer
            x2_name = f"layer_{layerIndex}_2_norm"
            x2 = nn.LayerNorm(eps=1e-6, normalized_shape=self.position_embedded_patches.size()[1:])
            self.transform_layers[x2_name] = x2

            # the multilayer perceptron
            x3_name = f"layer_{layerIndex}_mlp"
            x3 = nn.Sequential(
                nn.Linear(self.position_embedded_patches.size()[1], self.transformer_units[0]),
                nn.Linear(self.transformer_units[0], self.transformer_units[1]),
            )
            self.transform_layers[x3_name] = x3

            # the drop path TODO: what is a drop path??
            drop = DropPath(self.dropPathRate[layerIndex])
            self.transform_layers[f"layer_{layerIndex}_drop-path"] = drop
        self.last_norm = nn.LayerNorm(eps=1e-6, normalized_shape=self.position_embedded_patches.size()[1:])
        # create the mlp layer
        self.mlp_head = nn.Sequential()
        for layerIndex, units in enumerate(mlp_head_units):
            self.mlp_head.add_module(
                f"mlp_head_{layerIndex}",
                nn.Linear(units, units),
            )
            self.mlp_head.add_module(
                f"mlp_head_{layerIndex}_activation",
                nn.GELU(),
            )
            self.mlp_head.add_module(
                f"mlp_head_{layerIndex}_dropout",
                nn.Dropout(dropout_rate),
            )
        self.logits = nn.Linear(mlp_head_units[-1], activityCount)

    def init_weights(self) -> None:
        initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        # self.linear.bias.data.zero_()
        # self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        # generate the patches
        patches = self.patches(src)
        # add the position embedding to the patches
        position_embedded_patches = self.position_embedded_patches(patches)
        # now go throug the transformer layers
        encoded_patches = torch.tensor([])
        for layerIndex, moduleName in enumerate(self.transform_layers):
            # get the module
            print(moduleName, layerIndex)
            module = self.transform_layers[moduleName]
            # pass the input through the module
       # after going through all of the transformer layers, final normalization layer
        norm = self.last_norm(encoded_patches)
        # apply gap
        gap = nn.AvgPool1d(norm.size()[1])
        # pass throug the final multilayer perceptron
        mlp_head = self.mlp_head(gap)
        # pass through the logits
        logits = self.logits(mlp_head)
        return logits

class SensorPatches(nn.Linear):
    def __init__(self,projection_dim,patchSize,timeStep):
        """
                This applies 1D convolution to the input data to project it to the correct size
                1D Convolutional works by:
                1. Take input 1D array
                2. Multiply values in the array by a kernel of size kernel_size
                ex: kernel_size = 1
                your input is of size 12 and your kernel is of size 1 meaning that your output map will be of size 12
                your input is called the feature vector and the output is the feature map
                input 12x1 kernel 1x1 output 12x1
                [9,7,2,4,8,7,3,1,5,9,8,4], conv kernel = [6] -> [54,42,12,24,48,42,18,6,30,54,48,24] multiply by 6
                ex: kernel_size = 2
                12x1 kernel 2x1 output 11x1
               [9,7,2,4,8,7,3,1,5,9,8,4] -> [69,33,30,60,66,39,15,33,69,75,48] multiply 9*3 + 7*6 = 63
               [3,6] 2x1 kernel
               multiply 9*3 + 7*6 = 63 when kernel gets to [9,7,2,4,8,7,3,1,5,9,8,4]
                                                                               [3,6] you stop here
                                                                               (8*3)+(4*6) = 24+24 = 48
              padding can be used in a conv layer to make sure that the output is the same size as the input

            """
        super(SensorPatches).__init__()
        self.projection_dim = projection_dim
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.flatten = nn.Flatten()
        self.modal1_project = nn.Conv1d(out_channels=int(projection_dim/2), kernel_size=patchSize, stride=timeStep)
        self.modal2_project = nn.Conv1d(out_channels=int(projection_dim/2), kernel_size=patchSize, stride=timeStep)

    def forward(self, in_acc, in_gyro):
        # in_acc = self.flatten(in_acc) # if the data is not 1D then may need to flatten
        # in_gyro = self.flatten(in_gyro)
        acc = self.modal1_project(in_acc)
        gyro = self.modal2_project(in_gyro)
        projected = torch.cat((acc,gyro),dim=2) #combine the two modalities
        return projected

class PatchEncoder(nn.Linear):
    def __init__(self, num_patches:int, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, input_patch: Tensor) -> Tensor:
        positions = torch.arange(0, self.num_patches, 1) # create a tensor corresponding to position position
        encoded = input_patch + self.position_embedding(positions) # add the position embedding to the input patch
        return encoded


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, training=None):
        if(training):
            input_shape = x.shape
            batch_size = input_shape[0]
            ranking = x.shape.rank
            shape = (batch_size,) + (1,) * (ranking - 1)
            random_tensor = (1 - self.drop_prob) + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor = torch.floor(random_tensor)
            output = x.div(1 - self.drop_prob) * random_tensor
            return output
        else:
            return x
class SensorMultiHeadAttention(nn.Module):
    def __init__(self, projectionQuarter, num_heads, startIndex, stopIndex, dropout_rate=0.0, drop_path_rate=0.0):
        super(SensorMultiHeadAttention).__init__()
        self.projectionQuarter = projectionQuarter
        self.num_heads = num_heads
        self.start_index = startIndex
        self.stop_index = stopIndex
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.attention = nn.MultiheadAttention(embed_dim=projectionQuarter, num_heads=num_heads, dropout=dropout_rate)
        # pass input into the transformer attention attention  = torch.nn.MultiheadAttention(<input-size>, <num-heads>) -> x, _ = attention(x, x, x)
        self.drop_path = DropPath(drop_path_rate) #TODO figure out what a drop path is

    def forward(self, input, training=None, return_attention_scores=False):
        extractedInput = input[:, :, self.start_index:self.stop_index]
        if return_attention_scores:
            MHA_Outputs, attentionScores = self.attention(extractedInput, extractedInput, return_attention_scores=True)
            return MHA_Outputs, attentionScores
        else:
            MHA_Outputs = self.MHA(extractedInput, extractedInput)
            MHA_Outputs = self.DropPath(MHA_Outputs)
            return MHA_Outputs
