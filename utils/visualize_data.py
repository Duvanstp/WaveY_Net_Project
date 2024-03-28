import matplotlib.pyplot as plt
import seaborn as sns


from waveynet_model.Wave_Y_Net import *
from utils.import_data import *


path_train_local = r'C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\train_ds.npz'
path_test_local = r'C:\Folder_Personal\Física\Trabajo de grado\Data_WaveYNet\test_ds.npz'

train_structures, train_Hy_fields, train_dielectric_permittivities, test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = data_import(path_train_local, path_test_local)

sns.heatmap(train_Hy_fields[1,1,:,:], annot=False, cmap='coolwarm') # CMAP coolwarm H field, binary for structure
plt.show()



