## Importando librerias
import torch
import matplotlib.pyplot as plt
import numpy as np


from torch import nn
from tqdm import tqdm
from torchvision.transforms import ToTensor, Lambda, Compose

from utils.save_train_info import *

#### Aquí vamos a definir la estructura de la red UNET modificada como el esqueleto para construir la WaveY-Net
## IMPORTANTE: Pytorch toma las medidas diferente a tensorflow 
## (tamaño canal, x,y)

class Model_UNET(nn.Module):
    """
    #### Class: Model_UNET
    * Parametros: input_shape, lr=0.001
    * Description: It is the model general of the WaveY-Net
    """
    def __init__(self, learning_rate = 0.001):
        super(Model_UNET, self).__init__()
        self.learning_rate = learning_rate
        self.steps = [16,32,64,128,256,512]
        self.size_blocks = len(self.steps)
        self.size_capas = 6 
        self.Act = nn.LeakyReLU() 
        self.pool_Uniform = nn.MaxPool2d(kernel_size = (2, 2),
                                         stride = (2, 2),
                                         padding = 0)
        self.pool_NoUniform = nn.MaxPool2d(kernel_size = (1, 2),
                                         stride = (1, 2),
                                         padding = 0) 
        self.upsample = nn.Upsample(scale_factor=(2,2), mode='nearest')
        self.NoUniform_upsample = nn.Upsample(scale_factor=(1,2), mode='nearest')

        # el maxpoling no uniforme es [(1, 2), (1, 2), 0])] ORDEN kernel_size, strides, padding
        # para el maxpooling uniforme tenemos [(2, 2), (2, 2), 0])] ORDEN kernel_size, strides, padding

        ## Aquí vamos a crear los atributos, más reducido usando el setattr
        self.attributesE(self.steps)
    
    def attributesE(self, steps):
        """
        Parameters: A list of steps
        Description: Create attributes for each step.
        Returns: Attributes to encoder and decoder respectively
        """
        for step in steps:
            if  step == 16:
                setattr(self, f'B{step}EI', nn.Conv2d(1, step, (3,3), padding=1))
                setattr(self, f'B{step}DI', nn.Conv2d(step, 2, (3,3), padding=1))
            else:
                setattr(self, f'B{step}EI', nn.Conv2d(step//2, step, (3,3), padding=1))
                setattr(self, f'B{step}DI', nn.Conv2d(step, step//2, (3,3), padding=1))
            
            setattr(self, f'B{step}E', nn.Conv2d(step, step, (3,3), padding = 1))
            setattr(self, f'BN{step}', nn.BatchNorm2d(step))
            setattr(self, f'B{step}D', nn.Conv2d(step, step, (3,3), padding = 1)) 

    # Conexion residual
    def res_conection(self, capa1, capa2):
        '''
        Parameters: capa1, capa2
        Description: Residual connection
        Returns Add layers
        '''
        residual = capa1 + capa2
        return residual
    
    def CBA(self, Capa, BatchNorm, x, is_init = False):
        """
        Parameters: Capa, BatchNorm, x, is_init = False
        Description: Apply Convolucional, Batch normalization, and activacion layers if is_init is True just apply convolutional layer
        Returns: layer
        """
        if is_init:
            conv = Capa(x)
        else:
            conv = self.Act(BatchNorm(Capa(x)))
        return conv
    
    def Block(self, Capa, CapaI, BatchNorm, x):
        '''
        Parameters: Capa, CapaI, BatchNorm, x
        Description: Apply CBA  and residual conection six-times
        Returns: End layer
        '''
        c0 = self.CBA(CapaI, BatchNorm, x, is_init = True)
        c1 = self.CBA(Capa,BatchNorm, c0)
        c2 = self.CBA(Capa,BatchNorm, c1)
        c3 = self.CBA(Capa,BatchNorm, c2)
        c4 = self.CBA(Capa,BatchNorm, self.res_conection(c3,c1))
        c5 = self.CBA(Capa,BatchNorm, c4)
        c6 = self.CBA(Capa,BatchNorm, self.res_conection(c5,c4))
        return c6
    
    def cat(self, block1, block2):
        '''
        Parameters: block1, block2
        Description: Concatenate block1 and block2 in dim = 1
        Return: cat of block1, block2
        '''
        concatenate = torch.cat((block1, block2),dim = 1)
        return concatenate
    
    def conv_dec(self, size_1, size_2, x):
        '''
        Parameters: size_1, size_2, x
        Description: Apply convolutional layer with size_1 and size_2
        Returns: convolution of x
        '''
        conv = nn.Conv2d(size_1,size_2, (3,3), padding=1)
        return conv(x)
    
    def pre_conv_cat(self, block1, block2, upsample = True):
        '''
        Parameters: block1, block2, upsample.
        Description: Perform input preparation to a block with shortcut connection using uniform or non-uniform upsample (convolutional of block1 and upsampled block2)
        Return: Processed layer of block1, block2
        '''
        if upsample:
            size_1 = self.cat(block1, self.upsample(block2)).size()[1]
            size_2 = self.upsample(block2).size()[1]
            pre_conv = self.conv_dec(size_1, size_2, self.cat(block1, self.upsample(block2)))
        else:
            size_1 = self.cat(block1, self.NoUniform_upsample(block2)).size()[1]
            size_2 = self.NoUniform_upsample(block2).size()[1]
            pre_conv = self.conv_dec(size_1, size_2, self.cat(block1, self.NoUniform_upsample(block2)))
        return pre_conv

    def encoder(self,x):
        '''
        Parameters: X_train
        Description: Create encoder using Blocks function
        Return: layer for each block
        '''
        Block1E = self.Block(self.B16E, self.B16EI, self.BN16, x)
        Block2E = self.Block(self.B32E, self.B32EI, self.BN32, self.pool_NoUniform(Block1E))
        Block3E = self.Block(self.B64E, self.B64EI, self.BN64, self.pool_NoUniform(Block2E))
        Block4E = self.Block(self.B128E, self.B128EI, self.BN128, self.pool_Uniform(Block3E))
        Block5E = self.Block(self.B256E, self.B256EI, self.BN256, self.pool_Uniform(Block4E))
        Block6E = self.Block(self.B512E, self.B512EI, self.BN512, self.pool_Uniform(Block5E))
        return  Block1E,Block2E,Block3E,Block4E,Block5E,Block6E
    
    def decoder(self,x):
        '''
        Parameters
        --------------------------------
        X_train_batch: torch.Tensor
        --------------------------------
        Description: Create decoder using encoder and Block functions, It has shortcut conection and blocks of convolutions.
        Return: End layer of decoder
        '''
        Block1E,Block2E,Block3E,Block4E,Block5E,Block6E = self.encoder(x)
        Block5D = self.Block(self.B256D, self.B512DI, self.BN256, self.pre_conv_cat(Block5E, Block6E))
        Block4D = self.Block(self.B128D, self.B256DI, self.BN128, self.pre_conv_cat(Block4E, Block5D))
        Block3D = self.Block(self.B64D, self.B128DI, self.BN64, self.pre_conv_cat(Block3E, Block4D))
        Block2D = self.Block(self.B32D, self.B64DI, self.BN32, self.pre_conv_cat(Block2E, Block3D, upsample = False))
        Block1D = self.Block(self.B16D, self.B32DI, self.BN16, self.pre_conv_cat(Block1E, Block2D, upsample = False))
        END = self.CBA(self.B16DI, self.BN16, Block1D, is_init = True)
        return END
    def step_optimizer(self, X_train_batch, y_train_batch, loss_function, optimizer):
            """
            * This function 
            * Parameters:
            --------------------------------
            X_train_batch: torch.Tensor
            y_train_batch: torch.Tensor
            loss_function: nn.L1Loss
            optimizer: torch.optim.Adam
            
            * Returns: loss
            """
            outputs = self.decoder(X_train_batch)
            loss = loss_function(outputs, y_train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss
    def fit(self, X_train, y_train, epochs, batch_size):
        '''
        Parameters: X_train, y_train, epochs, batch_size
        Description: Train the model using the loss function and optimizer for batch.
        '''
        loss_function = nn.L1Loss()

        optimizer = torch.optim.Adam(self.parameters(),
                                    lr = self.learning_rate,
                                    weight_decay = 1e-8)
        losses = []

        train_batch = False
        if X_train.shape[0] > batch_size:
            train_batch = True
        
        num_batchs = X_train.shape[0]//batch_size
        num_batchs_rest = X_train.shape[0]%batch_size

        if train_batch:
            for epoch in range(epochs):
                with tqdm(total=num_batchs, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', ncols=50, bar_format='{l_bar}{bar:30}{r_bar} -> | Tiempo restante: {remaining}') as pbar_epoch:
                    for i in range(num_batchs):
                        X_train_batch = X_train[i*batch_size:(i+1)*batch_size]
                        y_train_batch = y_train[i*batch_size:(i+1)*batch_size]
                        loss = self.step_optimizer(X_train_batch, y_train_batch, loss_function, optimizer)
                        pbar_epoch.update(1)
                    if num_batchs_rest > 0:
                        X_train_batch = X_train[num_batchs*batch_size:]
                        y_train_batch = y_train[num_batchs*batch_size:]
                        loss = self.step_optimizer(X_train_batch, y_train_batch, loss_function, optimizer)
                        pbar_epoch.update(1)
                losses.append(loss)
        else:
            for epoch in range(epochs):
                with tqdm(total=num_batchs, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', ncols=50, bar_format='{l_bar}{bar:30}{r_bar} -> | Tiempo restante: {remaining}') as pbar_epoch:
                    X_train_batch = X_train
                    y_train_batch = y_train
                    loss = self.step_optimizer(X_train_batch, y_train_batch, loss_function, optimizer)
                    losses.append(loss)
                    pbar_epoch.update(1)
        return losses
    def predict(self, X_test):
        '''
        Parameters:
        -------------------------------- 
        X_test: torch.Tensor
        Description: Predict using the trained model
        Return: Predictions
        '''
        outputs = self.decoder(X_test)
        return outputs