import math

import numpy as np
import torch
from torch import nn, Tensor


class TransformerModel(nn.Module):

    def __init__(self, input_shape, activity_count: int, projection_dim = 192, patchSize = 16, timeStep = 16, num_heads = 3, filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31], mlp_head_units = [1024], dropout_rate = 0.3, useTokens = False):
        super().__init__()
        self.model_type = 'Transformer'

        self.projection_half = projection_dim // 2
        self.projection_quarter = projection_dim // 4
        self.dropPathRate = np.linspace(0, dropout_rate * 10, len(convKernels)) * 0.1
        self.transformer_units = [
            projection_dim * 2,
            projection_dim, ]
        #make an input layer that takes in the input and projects it to the correct size
        self.flatten = nn.Flatten() # flatten
        #TODO ADD BATCH SIZE
        batch_size = 32
        # self.layer_1 = nn.Linear(input_shape[0], out_features=activity_count)
        self.layer_1 = nn.Linear(6, out_features=6)
        self.patches = SensorPatches(projection_dim, patchSize, timeStep)
        self.patch_encoder = PatchEncoder(num_patches=patchSize, projection_dim=projection_dim)
        self.transform_layers = nn.ModuleDict() # add items as you go
        self.activation = nn.SiLU()
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



            """
                Since I am not using the liteformer as the third branch I will have each attention branch be half of the projection dimension
                because it is half accel and half gyro so each transformer branch will take 100% of each modality
            """

            acc_attention_name = f"layer_{layerIndex}_acc_attention"
            #acc attention branch
            acc_attention = SensorMultiHeadAttention(self.projection_half, num_heads,0,self.projection_half,drop_path_rate=self.dropPathRate[layerIndex],dropout_rate=dropout_rate)
            self.transform_layers[acc_attention_name] = acc_attention
            # gyro attention branch
            gyro_attention_name = f"layer_{layerIndex}_gyro_attention"
            gyro_attention = SensorMultiHeadAttention(self.projection_half, num_heads, self.projection_half,projection_dim, dropout_rate=dropout_rate, drop_path_rate=self.dropPathRate[layerIndex])
            self.transform_layers[gyro_attention_name] = gyro_attention

            # the next normalization layer
            x2_name = f"layer_{layerIndex}_2_norm"
            x2 = nn.LayerNorm(eps=1e-6, normalized_shape=projection_dim)
            self.transform_layers[x2_name] = x2

            # the multilayer perceptron
            x3_name = f"layer_{layerIndex}_mlp"
            # x3 = nn.Sequential(
            #     nn.Linear(in_features=activity_count, out_features=self.transformer_units[0]),
            #     nn.SiLU(),
            #     nn.Dropout(dropout_rate),
            #     nn.Linear(in_features=self.transformer_units[0], out_features=self.transformer_units[1]),
            # )
            x3 = MLP(activity_count, self.transformer_units, dropout_rate)
            self.transform_layers[x3_name] = x3

            # the drop path TODO: what is a drop path??
            drop = DropPath(self.dropPathRate[layerIndex])
            self.transform_layers[f"layer_{layerIndex}_drop-path"] = drop
        self.last_norm = nn.LayerNorm(eps=1e-6, normalized_shape=192)
        # create the mlp layer
        self.mlp_head = MLPHead(mlp_head_units, dropout_rate)
        # self.logits = nn.Linear(mlp_head_units[-1], activity_count)
        self.logits = nn.Linear(mlp_head_units[-1], activity_count) #TODO remove the magic numbers i put in to get everything running

    def init_weights(self) -> None:
        initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        # self.linear.bias.data.zero_()
        # self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        #src = src.cuda()
        # TODO watch out that you are doing the skip connection right and that only one vector is being normalized
        src = self.layer_1(src)
        # next apply the layer norm
        # src = self.flatten(src) # do i need to flatten?
        patches = self.patches(src)
        # apply the position embedding to the patches
        position_embedded_patches = self.patch_encoder(patches)
        # now go throug the transformer layers
        encoded_patches = position_embedded_patches
        past_output = (False, [])
        secondary_skip = None
        for layerIndex, module_name in enumerate(self.transform_layers):
            # get the module
            # print(module_name, layerIndex, encoded_patches.shape)
            module = self.transform_layers[module_name]
            if "attention" in module_name:
                #this is the case of a skip connection
                if layerIndex != len(self.transform_layers)-1 and not past_output[0]: # if not the last connection and it has not informed on skip connection
                    # mark as skip
                    out = module(position_embedded_patches)
                    past_output = (True, [out])
                elif past_output[0]:
                    # use the skip connection the
                    # the transformer model will return the output and the attention scores
                    out = module(position_embedded_patches)
                    past_output[1].append(out)
            elif past_output[0]: # you have the end of the skip connection
                # print('performing skip connection')
                # use the skip connection
                concat_attention = torch.cat(past_output[1], dim=2)
                # now add the skip connection to the encoded
                encoded_patches = encoded_patches + concat_attention
                # reset the skip connection
                past_output = (False, [])
                # apply the layer norm
                encoded_patches = module(encoded_patches)
                secondary_skip = encoded_patches # store this for after the dropout
            elif 'drop' in module_name and secondary_skip is not None:
                # apply the dropout
                encoded_patches = module(encoded_patches)
                # now add the drop path encoded with the secondary skip
                encoded_patches = encoded_patches + secondary_skip
                secondary_skip = None


            else:
                # apply the layer
                encoded_patches = module(encoded_patches)



       # after going through all of the transformer layers, final normalization layer
        norm = self.last_norm(encoded_patches)
        # apply gap
        gap = nn.AvgPool1d(norm.size()[1])(norm)
        # pass throug the final multilayer perceptron
        mlp_head = self.mlp_head(gap)
        # pass through the logits
        logits = self.logits(mlp_head)
        return logits


class SensorPatches(nn.Module):
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
        super(SensorPatches, self).__init__()
        self.projection_dim = projection_dim
        self.patchSize = patchSize
        self.timeStep = timeStep
        # self.flatten = nn.Flatten()
        # there are 3 channels in the sensors
        """
            Convolutional formula for output 
            O = ((W-K+2P)/S)+1
            O = output height/length
            W = input volume
            K = filter size
            P = padding
            S = stride 
            ex: input 12x1 kernel 2x1 output 11x1
            O = ((12-2+2*0)/1)+1 = 11
            
        """''
        # projection_dim/2 = 96 patchsize = 16 timestep = 16
        self.modal1_project = nn.Conv1d(in_channels=3, out_channels=int(projection_dim/2), kernel_size=patchSize, stride=timeStep)

        self.modal2_project = nn.Conv1d(in_channels=3,out_channels=int(projection_dim/2), kernel_size=patchSize, stride=timeStep)

    def forward(self, inp):
        # output from hart, they do not batch the data
        # accProjections: (None, 128, 3)
        # gyro: Tensor("model/sensor_patches/strided_slice_3:0", shape=(None, 128, 3), dtype=float32)

        # print('input to patches ', inp.shape)
        # in_acc = self.flatten(in_acc) # if the data is not 1D then may need to flatten
        # in_gyro = self.flatten(in_gyro)
        # inp = inp.unsqueeze(0)
        acc_data = inp[:,:,:3]
        gyro_dat = inp[:,:,3:]

        acc_data = acc_data.permute(0,2,1)
        gyro_dat = gyro_dat.permute(0,2,1)

        acc = self.modal1_project(acc_data) # pass in the first element of the input corresponding to the accelerometer up to 3
        # print('output of patches 1', acc.shape)
        gyro = self.modal2_project(gyro_dat)# pass in the last 3 channels corresponding to gyro from 3
        # print('output of patches 2', gyro.shape)
        projected = torch.cat((acc,gyro),dim=1) #combine the two modalities
        # print('output of patches ', projected.shape)
        return projected

class PatchEncoder(nn.Module):
    def __init__(self, num_patches:int, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = int(num_patches/2)
        self.projection_dim = projection_dim
        self.position_embedding = nn.Embedding(self.num_patches, projection_dim)

    def forward(self, input_patch: Tensor) -> Tensor:
        positions = torch.arange(0, self.num_patches, 1) # create a tensor corresponding to position position
        encoded = self.position_embedding(positions) # get the position embedding
        # encoded = input_patch +  encoded # add the position embedding to the input patch
        input_patch = input_patch.permute(0,2,1)
        encoded = input_patch + encoded
        return encoded


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, training=None):
        if(training):
            input_shape = x.shape
            batch_size = input_shape[0]
            ranking = len(input_shape) # the rank of the tensor is the number of dimensions it has
            shape = (batch_size,) + (1,) * (ranking - 1)
            random_tensor = (1 - self.drop_prob) + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor = torch.floor(random_tensor)
            output = x.div(1 - self.drop_prob) * random_tensor
            return output
        else:
            return x
class SensorMultiHeadAttention(nn.Module):
    def __init__(self, projection_half, num_heads, startIndex, stopIndex, dropout_rate=0.0, drop_path_rate=0.0):
        super(SensorMultiHeadAttention, self).__init__()
        self.projection_half = projection_half
        self.num_heads = num_heads
        self.start_index = startIndex # the starting index
        self.stop_index = stopIndex # the stop index of the data to process
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.attention = nn.MultiheadAttention(embed_dim=projection_half, num_heads=num_heads, dropout=dropout_rate)
        # pass input into the transformer attention attention  = torch.nn.MultiheadAttention(<input-size>, <num-heads>) -> x, _ = attention(x, x, x)
        self.drop_path = DropPath(drop_path_rate) #TODO figure out what a drop path is

    def forward(self, input, training=None, return_attention_scores=False):
        extractedInput = input[:, :, self.start_index:self.stop_index]
        if return_attention_scores:
            MHA_Outputs, attentionScores = self.attention(extractedInput, extractedInput)
            return MHA_Outputs, attentionScores
        else:
            MHA_Outputs,_ = self.attention(extractedInput, extractedInput, extractedInput)
            MHA_Outputs = self.drop_path(MHA_Outputs,training=True)
            return MHA_Outputs




class MLP(nn.Module):
    def __init__(self, activity_count, transformer_units, dropout_rate):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(in_features=192, out_features=transformer_units[0])
        self.layer2 = nn.SiLU()
        self.layer3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(in_features=transformer_units[0], out_features=transformer_units[1])

    def forward(self, x):
        #input dimensions = 32x8x192
        # x = x.permute(0,2,1)
        x = self.layer1(x) # 8x384     32*8 = 256
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # permute back to 32x192x8
        permuted = x.permute(0,2,1)
        return x



class MLPHead(nn.Module):
    def __init__(self, mlp_head_units, dropout_rate):
        super(MLPHead, self).__init__()

        self.mlp_head = nn.Sequential()
        for layerIndex, units in enumerate(mlp_head_units):
            # Linear layer
            self.mlp_head.add_module(
                f"mlp_head_{layerIndex}",
                nn.Linear(units, units), #1024, 1024
            )
            # GELU activation
            self.mlp_head.add_module(
                f"mlp_head_{layerIndex}_activation",
                nn.GELU(),
            )
            # Dropout
            self.mlp_head.add_module(
                f"mlp_head_{layerIndex}_dropout",
                nn.Dropout(dropout_rate),
            )

    def forward(self, x):
        print('in the final mlp head layer ', x.shape)
        # Print input shape before processing through layers
        # print("Input shape:", x.shape) # 32, 8, 24 input shape
        # the issue is 32*8 -> 256, 24

        # Forward pass through the MLP head
        for layer in self.mlp_head:
            # print(layer)
            x = layer(x)
            # Print output shape after each layer
            # print(f"Output shape after {layer.__class__.__name__}:", x.shape)

        return x