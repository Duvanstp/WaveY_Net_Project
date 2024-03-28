import io
import time
import requests
import numpy as np

def import_data_train(path_train):
    '''
    * Parameters: Path to train data
    * Description: This function imports train data
    * returns: train_structures, train_Hy_fields, train_dielectric_permittivities
    '''
    train_data = np.load(path_train)

    train_structures                = train_data['structures'] # (27000, 1, 64, 256)
    train_Hy_fields                 = train_data['Hy_fields'] # (27000, 2, 64, 256)
    train_dielectric_permittivities = train_data['dielectric_permittivities'] # (27000, 1, 1, 1)

    return train_structures, train_Hy_fields, train_dielectric_permittivities

def import_data_test(path_test):
    '''
    * Parameters: Path to test data
    * Description: This function imports test data
    * returns: test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities
    '''
    test_data = np.load(path_test)

    test_structures                = test_data['structures'] # (3000, 1, 64, 256
    test_Hy_fields                 = test_data['Hy_fields'] # (3000, 2, 64, 256)
    test_Ex_fields                 = test_data['Ex_fields'] # (3000, 2, 64, 256)
    test_Ez_fields                 = test_data['Ez_fields'] # (3000, 2, 64, 256)
    test_efficiencies              = test_data['efficiencies'] # (3000, 1, 1, 1)
    test_dielectric_permittivities = test_data['dielectric_permittivities'] # (3000, 1, 1, 1)

    return test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities

def request_data_train(path):
    '''
    * Parameters: Path to request train data
    * Description: This function request train data
    * returns: train_structures, train_Hy_fields, train_dielectric_permittivities
    '''
    print('Init request train')
    response = requests.get(path)
    response.raise_for_status()
    train_data = np.load(io.BytesIO(response.content))

    train_structures                = train_data['structures']
    train_Hy_fields                 = train_data['Hy_fields']
    train_dielectric_permittivities = train_data['dielectric_permittivities']
    print('Finish request train')

    return train_structures, train_Hy_fields, train_dielectric_permittivities

def request_data_test(path):
    '''
    * Parameters: Path to request test data   
    * Description: This function request test data
    * returns: test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities
    '''
    print('Init request test')
    response = requests.get(path)
    response.raise_for_status()
    test_data = np.load(io.BytesIO(response.content))

    test_structures                = test_data['structures']                  #shape: (3000, 1, 64, 256)
    test_Hy_fields                 = test_data['Hy_fields']                   #shape: (3000, 2, 64, 256)
    test_Ex_fields                 = test_data['Ex_fields']                   #shape: (3000, 2, 64, 256)
    test_Ez_fields                 = test_data['Ez_fields']                   #shape: (3000, 2, 64, 256)
    test_efficiencies              = test_data['efficiencies']                #shape: (3000, 1, 1, 1)
    test_dielectric_permittivities = test_data['dielectric_permittivities']   #shape: (3000, 1, 1, 1)

    print('Finish request test')
    return test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities

def data_import(path_train: str, path_test: str):
    '''
    * Parameters: path_train, path_test
    * Description: This function imports or requests the train and test data
    * returns: train_structures, train_Hy_fields, train_dielectric_permittivities, test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities
    '''
    try:
        train_structures, train_Hy_fields, train_dielectric_permittivities = import_data_train(path_train)
        test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = import_data_test(path_test)
    except Exception as e:
        print('Error al importar los datos locales iniciando request...')
        path_train_request = 'http://metanet.stanford.edu/static/search/waveynet/data/train_ds.npz'
        path_test_request = 'http://metanet.stanford.edu/static/search/waveynet/data/test_ds.npz'

        train_structures, train_Hy_fields, train_dielectric_permittivities = request_data_train(path_train_request)
        test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities = request_data_test(path_test_request)
    finally:
        return train_structures, train_Hy_fields, train_dielectric_permittivities, test_structures, test_Hy_fields, test_Ex_fields, test_Ez_fields, test_efficiencies, test_dielectric_permittivities

