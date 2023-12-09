import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from model import SensorPatches, PatchEncoder, DropPath, SensorMultiHeadAttention, MLP, MLPHead, process_module


class HartModel(nn.Module):
    def __init__(self, input_shape, activity_count: int, projection_dim=192, patchSize = 16, timeStep = 16, num_heads = 3, filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31], mlp_head_units = [1024], dropout_rate = 0.3, useTokens = False):
        super().__init__()
        self.model_type = 'Transformer'
        self.projection_dim = projection_dim
        self.projection_half = projection_dim // 2
        self.projection_quarter = projection_dim // 4
        self.activity_count = activity_count
        self.dropPathRate = np.linspace(0, dropout_rate * 10, len(convKernels)) * 0.1
        self.transformer_units = [
            projection_dim * 2,
            projection_dim
        ]
        self.input = nn.Linear(input_shape[1], out_features=input_shape[1])
        self.patches = SensorPatches(projection_dim, patchSize, timeStep, num_modal=2) #The HART model assumes that there only two modalities
        self.patch_encoder = PatchEncoder(num_patches=patchSize, projection_dim=self.projection_dim)
        self.transform_layers = nn.ModuleDict() # add items as you go
        self.activation = nn.SiLU()
        self.dropOut = nn.Dropout(dropout_rate)
        self.mlp_head_units = mlp_head_units

        for layerIndex, kernelLength in enumerate(convKernels):
            first_layer_name = f"layer_{layerIndex}_norm-0"
            x1 = nn.LayerNorm(eps=1e-6, normalized_shape=self.projection_dim) # the get the dimensions of the input except the batch size
            self.transform_layers[first_layer_name] = x1

            first_attention_branch = f"liteformer_{layerIndex}_attention"
            branch_one = LiteFormer(startIndex=self.projection_quarter, stopIndex=self.projection_quarter+self.projection_half,
            projectionSize=self.projection_half, attentionHead=filterAttentionHead, kernelSize=kernelLength,dropPathRate=self.dropPathRate[layerIndex],
            dropout_rate=dropout_rate)
            self.transform_layers[first_attention_branch] = branch_one

            acc_attention_branch = f"acc_{layerIndex}_attention"
            branch_acc = SensorMultiHeadAttention(self.projection_quarter, num_heads,0,self.projection_quarter, drop_path_rate=self.dropPathRate[layerIndex], dropout_rate=dropout_rate)
            self.transform_layers[acc_attention_branch] = branch_acc

            gyro_attention_branch = f"gyro_{layerIndex}_attention-end"
            branch_gyro_attention = SensorMultiHeadAttention(self.projection_quarter, num_heads, self.projection_quarter+self.projection_half, projection_dim, drop_path_rate=self.dropPathRate[layerIndex],dropout_rate=dropout_rate)
            self.transform_layers[gyro_attention_branch] = branch_gyro_attention


            x2_name = f"layer_{layerIndex}_2_norm"
            x2 = nn.LayerNorm(eps=1e-6, normalized_shape=self.projection_dim)
            self.transform_layers[x2_name] = x2

            x3_name = f"layer_{layerIndex}_mlp"
            x3 = MLP(self.projection_dim, self.transformer_units, dropout_rate)
            self.transform_layers[x3_name] = x3

            drop = DropPath(self.dropPathRate[layerIndex])
            self.transform_layers[f"layer_{layerIndex}_drop-path"] = drop

        self.last_norm = nn.LayerNorm(eps=1e-6, normalized_shape=self.projection_dim)
        # create the mlp layer
        self.mlp_head = MLPHead(self.projection_dim, mlp_head_units, dropout_rate)
        self.classification = nn.Linear(mlp_head_units[-1], activity_count)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, src):
        src = self.input(src)
        # next apply the layer norm
        # src = self.flatten(src) # do i need to flatten?
        patches = self.patches(src)  # this outputs 3x8x192 in keras
        # apply the position embedding to the patches
        position_embedded_patches = self.patch_encoder(patches)  # this ouputs 32x8x192 in keras
        # now go throug the transformer layers
        encoded_patches = position_embedded_patches
        branch_ouputs = {
            "encoded_inputs": encoded_patches,
        }
        for layerIndex, module_name in enumerate(self.transform_layers):
            code = module_name[module_name.rindex("_"):]
            process_module(code, self.transform_layers[module_name], branch_ouputs)
        encoded_patches = branch_ouputs["encoded_inputs"]
        # print('encoded patches ', encoded_patches.shape)
        # after going through all of the transformer layers, final normalization layer
        norm = self.last_norm(
            encoded_patches)  # 32,8,192 i think it should be the other way around? theirs is 8x192 as well which should probably be swapped in our case
        # apply gap
        # gap = nn.AvgPool1d(norm.size()[1])(norm) # this needs to go to 32x192
        # gap = nn.AdaptiveAvgPool1d(1)(norm).squeeze()
        gap = norm.mean(dim=1)
        # pass throug the final multilayer perceptron
        mlp_head = self.mlp_head(gap)
        # pass through the logits
        logits = self.classification(mlp_head)
        return self.softmax(logits)

class LiteFormer(nn.Module):
    def __init__(self,startIndex,stopIndex, projectionSize, kernelSize = 16, attentionHead = 3, use_bias=False, dropPathRate = 0.0,dropout_rate = 0,**kwargs):
        super(LiteFormer, self).__init__(**kwargs)
        self.use_bias =  use_bias
        self.start_index = startIndex
        self.stop_index = stopIndex
        self.kernel_size = kernelSize
        self.softmax = nn.Softmax()
        self.projection_size = projectionSize
        self.attention_head = attentionHead
        self.DropPathLayer = DropPath(dropPathRate)
        self.projection_half = projectionSize //2
        self.depthwise_kernel = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernelSize, bias=self.use_bias, padding='same', groups=1)
            for _ in range(self.attention_head)
        ])
        self.init_weights()


    def init_weights(self):
        for conv_layer in self.depthwise_kernel:
            #init the weights of the conv layer
            init.xavier_uniform_(conv_layer.weight.data)

            if conv_layer.bias is not None:
                init.zeros_(conv_layer.bias.data)


    def forward(self, x, train=None):
        input_vals = x[:,:,self.start_index:self.stop_index]
        input_shape = input_vals.shape
        # I may have to permute for the conv layers
        # input_data = x.permute(0,2,1) #TODO: do i need to do ?
        input_data = input_vals.reshape(-1,self.attention_head, input_shape[1]) # this reshape gets the data in the format N, H, channel, W no permute needed

        # input_data = reshaped_input.permute(0, 1, 3,2) #TODO: do i need to do ? flip the channel with the sequence length gives the format batch height width channel

        #apply the softmax on the conv kernels
        if train:
            for conv_idx in range(self.attention_head):
                self.depthwise_kernel[conv_idx].weight.data = nn.functional.softmax(self.depthwise_kernel[conv_idx].weight.data, dim=0)

        #get the outputs

        convolution_outputs = torch.cat([
            nn.functional.conv1d(input_data[:,conv_idx:conv_idx+1, :],
                          self.depthwise_kernel[conv_idx].weight,
                          stride=1,
                          padding=self.kernel_size//2)
            for conv_idx in range(self.attention_head)
        ], dim=1)
        conv_outputs_drop = self.DropPathLayer(convolution_outputs)

        local_att = conv_outputs_drop.reshape(input_shape[0], input_shape[1], -1)
        # unpermute = local_att.permute(0,2,1)
        return local_att





