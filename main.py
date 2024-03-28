import matplotlib.pyplot as plt
import seaborn as sns
import time
import os


from waveynet_model.Wave_Y_Net import *
from utils.import_data import data_import

def main(sample_size, batch_size, epochs, device, path_train_local, path_test_local, base_dir_save):
    """
    #### Main function
    This function is the main function of the project.
    It is responsible for the execution of the project.
    """
    print('Iniciando')
    init_time = time.time()

    train_structures, train_Hy_fields, train_dielectric_permittivities, test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = data_import(path_train_local, path_test_local)

    WaveYnet = Model_UNET(learning_rate=0.001).to(device=device)

    X_train = torch.tensor(train_structures[0:sample_size,:,:,:], dtype = torch.float)
    y_train = torch.tensor(train_Hy_fields[0:sample_size,:,:,:], dtype = torch.float)
    
    losses = WaveYnet.fit(X_train, y_train, epochs, batch_size)

    losses = [loss.item() for loss in losses]

    parameters = WaveYnet.parameters()
    
    num_parametros = sum(p.numel() for p in WaveYnet.parameters())
    print('El numero de parametros de la red es de: ',num_parametros)

    torch.save(WaveYnet.state_dict(), os.path.join(os.getcwd() , r'weights_model\weights2.pth'))

    # weights1 300 samples 16 batch, 50 epochs
    # WaveYnet.load_state_dict(torch.load('pesos_modelo.pth')) # cargar pesos de nuevo

    end_time = time.time()
    training_time = (end_time - init_time)/60

    create_json(epochs, losses, training_time, None, sample_size, 3, base_dir_save)
    print('Finalizado')
    print('Tiempo empleado: ', round(training_time, 3) , ' minutos')

    return losses, epochs

if __name__ == "__main__":
    base_path = os.getcwd()
    base_path_exit = os.path.dirname(base_path)
    path_train_local = os.path.join(base_path_exit, r'Data_WaveYNet\train_ds.npz')
    path_test_local = os.path.join(base_path_exit, r'Data_WaveYNet\test_ds.npz')
    base_dir_save = os.path.join(base_path, r'results_json')

    sample_size = 1
    batch_size = 6
    epochs = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using {} device'.format(device))

    losses, epochs = main(sample_size, batch_size,epochs, device, path_train_local, path_test_local, base_dir_save)

    sns.lineplot(
        x = list(range(epochs)), 
        y = losses
    )
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.grid()
    plt.show()