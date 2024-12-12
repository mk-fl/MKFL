from pathlib import Path
import sys
import os
cwd=os.getcwd()
sys.path.insert(0, cwd)
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")
sys.path.append('../../../../../TenSEAL')
import torch
import  SemaClassifier.classifier.GNN.gnn_helpers.metrics_utils as metrics_utils
import SemaClassifier.classifier.GNN.GNN_script as GNN_script
from SemaClassifier.classifier.Images import ImageClassifier  as img

import main_utils
import parse_args

import flwr as fl
from AESCipher import AESCipher
import secrets
import string
import rsa

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
import copy
import time


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
EPOCHS=5
BATCH_SIZE_TEST=32

class GNNClient(fl.client.NumPyClient):
    """Flower client implementing Graph Neural Networks using PyTorch."""

    def __init__(self, m, dataset,y_test,id,model_type,pk=None,filename=None, dirname=None ) -> None:
        super().__init__()
        self.m = m
        self.d =dataset
        self.y_test=y_test
        self.id=id
        self.round=0
        self.model_type=model_type
        self.global_model = m.model
        #for storing results
        self.filename = filename
        self.dirname = dirname
        self.train_time = 0
        #for RSA encryption
        self.publickey = ""
        
    
    def set_public_key(self, rsa_public_key):
        self.publickey = rsa.PublicKey(int(rsa_public_key[0]),int(rsa_public_key[1]))
        return
  
    def get_parameters(self, config: Dict[str, str]=None) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.m.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.m.model.train()
        params_dict = zip(self.m.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.m.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config:Dict[str,str]) -> Tuple[List[np.ndarray], int, Dict]:
        if "wait" in config and config["wait"] == "no_update":
                pass
        else:
            self.set_parameters(parameters)
        self.global_model = copy.deepcopy(self.m.model)
        torch.save(self.global_model,f"{self.dirname}/model_global_{self.round}.pt")
        self.round+=1
        test_time, loss, y_pred = self.m.test(self.m.model,self.d.testset,BATCH_SIZE_TEST,self.id,device=DEVICE)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        m,loss=self.m.train(self.m.model,self.d.trainset,EPOCHS,BATCH_SIZE,self.id,device=DEVICE)
        torch.save(self.m.model,f"{self.dirname}/model_local_{self.round}.pt")
        self.train_time=loss["train_time"]
        l=loss["loss"]
        main_utils.cprint(f"Client {self.id}: Evaluation accuracy & loss, {l}, {acc}, {prec}, {rec}, {f1}, {bal_acc}", self.id)
        metrics_utils.write_to_csv([str(self.m.model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss["loss"], self.train_time, test_time, str(np.array_str(np.array(y_pred),max_line_width=10**50))], self.filename)

        return self.get_parameters(config={}), len(self.d.trainset) ,{"accuracy": float(acc),"precision": float(prec), "recall": float(rec), "f1": float(f1), "balanced_accuracy": float(bal_acc),"loss": float(loss["loss"]),"test_time": float(test_time),"train_time":float(self.train_time)}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        #self.set_parameters(parameters)
        test_time, loss, y_pred = self.m.test(self.m.model,self.d.testset,BATCH_SIZE_TEST,self.id,device=DEVICE)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        metrics_utils.write_to_csv([str(self.m.model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss, self.train_time, test_time, str(np.array_str(np.array(y_pred),max_line_width=10**50))], self.filename)
        main_utils.cprint(f"Client {self.id}: Evaluation accuracy & loss, {loss}, {acc}, {prec}, {rec}, {f1}, {bal_acc}", self.id)
        return float(loss), len(self.d.testset), {"accuracy": float(acc),"precision": float(prec), "recall": float(rec), "f1": float(f1), "balanced_accuracy": float(bal_acc),"loss": float(loss),"test_time": float(test_time),"train_time":float(self.train_time)}

    def get_gradients(self):
        parameters = [np.array([self.id])] + [val.cpu().numpy() for _, val in self.m.model.state_dict().items()]
        characters = string.ascii_letters + string.digits + string.punctuation
        AESKEY = ''.join(secrets.choice(characters) for _ in range(245))
        encrypted_parameters = AESCipher(AESKEY).encrypt(parameters)
        encrypted_key = rsa.encrypt(AESKEY.encode('utf8'), self.publickey)
        return encrypted_key + encrypted_parameters
    
    
def main() -> None:

    # Parse command line arguments
    n_clients, id, filename, dataset_name, model_path, model_type, datapath, split = parse_args.parse_arg_client()

    dirname = None
    if filename is not None:
        timestr1 = time.strftime("%Y%m%d-%H%M%S")
        timestr2 = time.strftime("%Y%m%d-%H%M")
        dirname = f"{filename}/{timestr2}_wo/parms_{id}/"
        filename = f"{filename}/{timestr2}_wo/client{id}_{timestr1}.csv"
    print("FFFNNN",filename)

    if not os.path.isdir(dirname):
        os.makedirs(os.path.dirname(dirname), exist_ok=True)

    #Dataset Loading
    #Modify the init_datasets function in main_utils
    d= main_utils.init_datasets(dataset_name, datapath, split, n_clients, id)
    main_utils.cprint(f"Client {id} : datasets length, {len(d.trainset)}, {len(d.testset)}",id)
    

    #Model
    #Modify the get_model function in main_utils
    m = main_utils.get_model(model_type, d.classes, d.trainset, model_path=model_path)

    #Client
    client = GNNClient(m, d,d.y_test,id,model_type, filename=filename, dirname=dirname)
    #torch.save(model, f"HE/GNN_model.pt")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client, root_certificates=Path("./FL/.cache/certificates/ca.crt").read_bytes())
    with open(filename,'a') as f:
        f.write(str(d.y_test)+"\n")
    return filename
    
if __name__ == "__main__":
    main()
    
