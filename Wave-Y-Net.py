## Importando librerias
import torch
import matplotlib.pyplot as plt
import numpy as np


from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.nn import Flatten, Sequential, Linear, ReLU


## Verificamos si tenemos acceso a la gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using {} device'.format(device))

#### Aquí vamos a definir la estructura de la red UNET modificada como el esqueleto para construir la WaveY-Net
## IMPORTANTE: Pytorch toma las medidas diferente a tensorflow 
## (tamaño canal, x,y)

class Model_UNET(nn.Module):
    def __init__(self, input_shape, lr = 0.001):
        super(Model_UNET, self).__init__()
        """
        Parametros: input_shape, lr=0.001
        """
        #definimos los bloques de down-up de la UNET
        self.steps = [16,32,64,128,256,512]
        self.size_blocks = 6
        self.size_capas = 6
        self.Act = nn.LeakyReLU()
        self.pool_Uniform = nn.MaxPool2d(kernel_size = (2, 2),
                                         stride = (2, 2),
                                         padding = 0)
        self.pool_NoUniform = nn.MaxPool2d(kernel_size = (1, 2),
                                         stride = (1, 2),
                                         padding = 0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        ## Aquí vamos a crear los atributos, más reducido usando el seattr
        for step in self.steps:
            if  step == 16:
                setattr(self, f'B{step}EI', nn.Conv2d(1, step, (3,3), padding=1))
            else:
                setattr(self, f'B{step}EI', nn.Conv2d(step//2, step, (3,3), padding=1))
            setattr(self, f'B{step}E', nn.Conv2d(step, step, (3,3), padding = 1))
            setattr(self, f'BN{step}', nn.BatchNorm2d(step))
    
    # va a ser la conexion residual probar si con imagenes funciona bien
    def res_conection(self, capa1, capa2):
        residual = capa1 + capa2
        return residual
    
    # Bloque convolucional, normalization y activation, si es el inicial no lleva activacion
    def CBA(self, Capa, BatchNorm, x, is_init = False):
        """
        Capa convolucional,
        Batch normalization 
        y activacion
        """
        if is_init:
            conv = Capa(x)
        else:
            conv = self.Act(BatchNorm(Capa(x)))
        return conv
    
    def Block(self, Capa, CapaI, BatchNorm, x):
        c0 = self.CBA(CapaI, BatchNorm, x, is_init = True) # revisar al parecer no lleva activation
        c1 = self.CBA(Capa,BatchNorm, c0)
        c2 = self.CBA(Capa,BatchNorm, c1)
        c3 = self.CBA(Capa,BatchNorm, c2)
        c4 = self.CBA(Capa,BatchNorm, self.res_conection(c3,c1))
        c5 = self.CBA(Capa,BatchNorm, c4)
        c6 = self.CBA(Capa,BatchNorm, self.res_conection(c5,c4))
        return c6
        
    def encoder(self,x): # Nota hace falta el maxpoling no uniforme
        Block1 = self.Block(self.B16E, self.B16EI, self.BN16, x)
        Block2 = self.Block(self.B32E, self.B32EI, self.BN32, Block1)
        Block3 = self.Block(self.B64E, self.B64EI, self.BN64, Block2)
        Block4 = self.Block(self.B128E, self.B128EI, self.BN128, Block3)
        Block5 = self.Block(self.B256E, self.B256EI, self.BN256, Block4)
        Block6 = self.Block(self.B512E, self.B512EI, self.BN512, Block5)
        Blocks = [Block1,Block2,Block3,Block4,Block5,Block6]
        return  Blocks # Nota: la idea es poder usar cada una de las salidas de los bloques
        # Nota importante no he hecho el pooling no uniforme a las capas.
        # podriamos probar pooling normal(average, maxpooling) a ver que sucede
        # el maxpooling está reduciendo a la mitad las imagenes.
        # se realiza un upsampling para agrandar la imagen en el decoder
        # el maxpoling no uniforme es [(1, 2), (1, 2), 0])]
        # orden kernel_size, strides, padding
        # para el maxpooling uniforme tenemos [(2, 2), (2, 2), 0])]
        # con el mismo orden.
    
    def decoder(self,x):
        pass #Idealmente debe usar los bloques que salen del encoder
    
    def grad_step(self):
        pass
    
    def optimizer(self):
        pass
    
    def fit(self):
        pass
    

        
WaveY = Model_UNET(input_shape = (1,64,64))
WaveY