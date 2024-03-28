import os
import json


def save_json_info(json_info, base_dir, num):
    """
    * Save information in folder local
    * Parameters:
    --------------------------------
    json_info :  dictionary
    base_dir : string  
    num : int
    * Description: This function saves the train info in a json file
    * returns: None
    """
    name_file = "train" + str(num) + ".json"
    path = os.path.join(base_dir, name_file)
    with open(path, 'w') as outfile:
        json.dump(json_info, outfile)

def load_json_info(base_dir: str, num: int):
    """
    * Load information from folder local
    * Parameters:
    --------------------------------
    base_dir : string  
    num : int
    * Description: This function loads the train info from a json file
    * returns: dictionary
    """
    name_file = "train" + str(num) + ".json"
    path = os.path.join(base_dir, name_file)
    with open(path, 'r') as infile:
        json_info = json.load(infile)
    return json_info

def create_json(num_epochs, loss, time_train, accuracy, sample_size, num, base_dir):
    """
    --------------------------------
    Parameters:
    --------------------------------
        num_epochs : int
        loss : List[float]
        time_train : float
        accuracy : float
        sample_size : int
        num : int
        base_dir : string
    --------------------------------
    Description: 
        This function saves the train info in a json file
    --------------------------------
    Returns:
        json_info : dictionary
    """
    json_info = {
        "num_epochs": num_epochs,
        "loss": loss,
        "time_train [min]": time_train,
        "accuracy": accuracy,
        "sample_size": sample_size
    }
    save_json_info(json_info, base_dir, num)

    return json_info
