import numpy as np

#Load the train data
train_data = np.load('/your/local/path/train_ds.npz')

train_structures                = train_data['structures']
train_Hy_fields                 = train_data['Hy_fields']
train_dielectric_permittivities = train_data['dielectric_permittivities']

#Load the test data
test_data = np.load('/your/local/path/test_ds.npz')

test_structures                = test_data['structures']
test_Hy_fields                 = test_data['Hy_fields']
test_Ex_fields                 = test_data['Ex_fields']
test_Ez_fields                 = test_data['Ez_fields']
test_efficiencies              = test_data['efficiencies']
test_dielectric_permittivities = test_data['dielectric_permittivities']