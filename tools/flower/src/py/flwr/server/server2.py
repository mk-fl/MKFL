# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""


import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union
import time
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
    serde,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

import sys
sys.path.append('../../../../../TenSEAL')
import tenseal as ts
import numpy as np
from functools import reduce
import random

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]

def reshape_parameters(parameters,shapes):
    p=[]
    offset=0
    for _,s in enumerate(shapes):
        n = int(np.prod(s))
        if not s:
            p.append(parameters[offset])#np.array(parameters[offset]))#,dtype=object))
        else:
            p.append(list(np.array(parameters[offset:(offset+n)]).reshape(s)))
        offset+=n
    return p#np.array(p,dtype=object)

class Server:
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        contribution: bool = True,
        strategy: Optional[Strategy] = None,
        methodo, 
        threshold,
        shapes=None,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None
        self.context = ts.context(ts.SCHEME_TYPE.MK_CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.context.global_scale = 2**40
        self.pk = None
        self.n = 0
        self.n_tot = 1
        self.length=0
        self.clients = None
        self.client_mapping = None
        self.ce = contribution
        self.shape = shapes
        self.methodo = methodo
        self.threshold = threshold
        self.waiting = []
        self.check = True

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager
    
    def get_pk(self, timeout: Optional[float]):
        clients = self.clients
        if clients is None:
            min_num_clients = self.strategy.min_available_clients
            #TODO check if all clients are available
            clients = self._client_manager.sample_cid(min_num_clients,min_num_clients)
        get_pk_ins = self.context
        client_instructions= [(client, (get_pk_ins, cid)) for cid,client in clients]
        # Get public keys from all clients
        results, failures = fn_clients(
            client_instructions=client_instructions,
            client_fn=get_pk_client,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        if len(failures)!=0 :
            raise RuntimeError("Error while getting the public keys")
        #Aggregate public keys
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate(0, results[1:], lambda x,y : x.add_pk(y[1]),results[0][1], failures)
        parameters_aggregated, metrics_aggregated = aggregated_result
        if parameters_aggregated is None :
            raise RuntimeError("Error while aggregating public keys")
        return parameters_aggregated, len(results),[client for c,client in clients] #clients
    
    def set_pk(self, pk):
        self.context.data.set_publickey(ts._ts_cpp.PublicKey(pk.data.ciphertext()[0]))
        self.pk = self.context.public_key()
    
    def send_pk(self, ctx, timeout : Optional[float]):
        clients = self.clients
        if clients is None:
            min_num_clients = self.strategy.min_available_clients        
            clients = self._client_manager.sample(min_num_clients,min_num_clients)
            #Send aggregated public key to all clients
        send_pk_ins = ctx
        client_instructions= [(client, send_pk_ins) for client in clients]
        results, failures = fn_clients(
            client_instructions=client_instructions,
            client_fn=send_pk_client,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        if len(failures)!=0:
            raise RuntimeError("Error while sending the public key")
        for _, result in results:
            if result != "ok":
                raise RuntimeError("Error while sending the aggregated public key")
        return
    
    def get_initial_parameters_enc(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from all of the available clients."""
        clients = None #self.clients
        if clients is None:
            min_num_clients = self.strategy.min_available_clients  
            clients = self._client_manager.sample_cid(min_num_clients,min_num_clients)
            
        # Get initial parameters from all clients
        client_instructions= [(client, cid) for cid,client in clients]
        results, failures = fn_clients(
            client_instructions=client_instructions,
            client_fn=get_parms_client,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        if len(failures)!=0 or len(results)!=self.n:
            raise RuntimeError("Error while getting initial parameters")
        # Aggregate initial parameters
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate(0, results[1:], lambda x,y : x.add(y[1]),results[0][1], failures)
        parameters_aggregated, metrics_aggregated = aggregated_result
        if parameters_aggregated is None or len(failures)!=0:
            raise RuntimeError("Error while getting initial parameters")
        #parameters_aggregated= parameters_aggregated*(1/self.n)
        return parameters_aggregated

    def compute_reputation(self, server_round, timeout):
        clients = self.clients + self.waiting
        client_instructions= [(client, None) for client in clients]
        results, failures = fn_clients(
            client_fn=get_gradients,
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "gradients %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
        clients_order = [r[0] for r in results]
        gradients = [r[1] for r in results]
        ce_server = self._client_manager.ce_server
        client_instructions= [(ce_server, gradients)]
        results, failures = fn_clients(
            client_fn=get_contributions,
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "contributions %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
        sv = results[0][1]
        return {clients_order[i]:sv[i] for i in range(len(sv))}
        
    def update_n(self):
        self.n = len(self.clients)
        self.strategy.min_fit_clients = self.n
        self.strategy.min_evaluate_clients = self.n
        self.strategy.min_available_clients = self.n
        return 
        
    def set_aside_clients(self, SVs, timeout):
        changed = False
        if SVs[-1][0] not in self.clients:
            self.clients = self._client_manager.reregister(SVs[-1][0])
            self.update_n()
            changed = True
        if SVs[-2][0] not in self.clients:
            self.clients = self._client_manager.reregister(SVs[-2][0])
            self.update_n()
            changed = True
        for sv in SVs:
            if sv[0] not in self.clients and sv[1] > self.threshold:
                self.clients = self._client_manager.reregister(sv[0])
                self.update_n()
                changed = True
        for sv in SVs:
            if len(self.clients) > 2 and sv[0] in self.clients and sv[1] < self.threshold:
                self.clients = self._client_manager.set_aside(sv[0])
                self.update_n()
                changed = True
        return changed
                
    def eliminate_clients(self, shapley_values,server_round, timeout):
        sorted_shapley_values = sorted(shapley_values.items(), key=lambda x:x[1])
        if len(shapley_values) > 2 and sorted_shapley_values[0][1] < self.threshold:
            if self.methodo == "delete_one":
                self.clients = self._client_manager.eliminate(sorted_shapley_values[0][0])
                self.update_n()
                return True
            elif self.methodo == "delete":
                sv = 0
                while self.n > 2 and sorted_shapley_values[sv][1] < self.threshold:
                    self.clients = self._client_manager.eliminate(sorted_shapley_values[sv][0])
                    self.update_n()
                    sv += 1
                return True
            elif self.methodo in ["set_aside","set_aside2"]:
                changed = self.set_aside_clients(sorted_shapley_values, timeout)
                if changed:
                    self.waiting = list(self._client_manager.waiting.values())
                return changed
            else:
                return False
        return False
        
    def fit_round_enc(
        self,
        server_round: int,
        current_round: int,
        start_time: float,
        history: History,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        clients = self.clients
        client_instructions = self.strategy.configure_fit_enc(
            (self.context,self.parameters) ,
            client_manager=self._client_manager, 
            clients = clients
        )
        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Send Encrypted parameters to all clients participating in this round
        # Collect `decryption shares` from all clients participating in this round
        results, failures = fn_clients(
            client_fn=send_enc,
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
        if len(failures) != 0 or len(results) != self.n:
            raise RuntimeError("Error while getting the decryption shares")
        
        # Aggregate decryption shares
        aggregated_ds: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate(server_round, results, lambda x,y : x.add_share(y[1]),self.parameters,failures)
        
        parameters_aggregated, metrics_aggregated = aggregated_ds[0]*(1/self.n), aggregated_ds[1]#*(1/self.n)
        # Evaluate model using strategy implementation
        self.evaluate_enc(parameters_aggregated,server_round-1,start_time,history)

        #Send aggregated parameters to all clients
        #Collects the new parameters from all clients
        #client_instructions2 = self.strategy.configure_fit_enc(
        #    (self.context,parameters_aggregated) ,
        #    client_manager=self._client_manager,
        #    clients = clients
        #)
        p2=ndarrays_to_parameters(reshape_parameters(parameters_aggregated.mk_decode(),self.shape))
        client_instructions2 = self.strategy.configure_fit(
            server_round=server_round,
            parameters=p2 ,
            client_manager=self._client_manager,
        )

        if not client_instructions2:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions2),
            self._client_manager.num_available(),
        )

        if self.ce and self.methodo == "set_aside":
            fit_ins = (self.context,ts.ckks_vector(self.context,np.array([0],dtype=object).flatten()))
            client_instructions2 += [(client, fit_ins) for client in self.waiting]
            log(INFO, "set_aside: " + str(len(self.clients)) + " active clients and " + str(len(self.waiting)) + " waiting clients")
        elif self.ce and self.methodo == "set_aside2":
            fit_ins = (self.context,parameters_aggregated)
            client_instructions2 += [(client, fit_ins) for client in self.waiting]
            log(INFO, "set_aside2: " + str(len(self.clients)) + " active clients and " + str(len(self.waiting)) + " waiting clients")

        # Collect `fit` results from all clients participating in this round
        results2, failures2 = fn_clients(
            client_fn=send_ds,
            client_instructions=client_instructions2,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results2),
            len(failures2),
        )
        results2 = [x for x in results2 if x[0] in self.clients]
        if len(failures2) != 0 or len(results2) != self.n:
            raise RuntimeError("Error while getting the new parameters")
            self.change_n(len(results2),[c for c,_ in results2] ,timeout)
            
        #send aggregated parameters to ce server
        if self.ce and self.check:
            instruction = EvaluateIns(parameters=ndarrays_to_parameters(parameters_aggregated.mk_decode()),config={})
            evaluate_client(self._client_manager.ce_server, instruction, timeout)
            self.check = False
            
        # compute the reputation of each client
        self.check = not random.randrange(5) #evaluate contribution with a probability of 20%
        if self.ce and self.check:
            log(INFO, "#################CONTRIBUTION EVALUATION#################")
            shapley_values = self.compute_reputation(server_round, timeout)
            log(INFO, "Shapley values round " + str(server_round) + " : " + str([(self.client_mapping[x.cid], shapley_values[x]) for x in shapley_values]))
            changed = self.eliminate_clients(shapley_values,server_round, timeout)
            if changed:
                self.change_n(self.n,self.clients,timeout)
                return None
                
        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit_enc(server_round, results2[1:], lambda x,y : x.add(y[1][0]),results2[0][1], failures2) #**y[1][1].num_examples
        parameters_aggregated, metrics_aggregated, self.n_tot = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)



    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float],length) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()
        # distributed : weighted average of clients results with new parameters
        # distributed_fit : weighted average of clients results with received parameters
        # centralized : server result

        if self.ce:
            log(INFO, "Identify the CE server")
            min_num_clients = self.strategy.min_available_clients
            clients = self._client_manager.sample(min_num_clients+1,min_num_clients+1,timeout = 300)
            if clients == []:
                return history
            client_instructions= [(client, None) for client in clients]
            
            results, failures = fn_clients(
                client_fn=identify,
                client_instructions=client_instructions,
                max_workers=self.max_workers,
                timeout=timeout,
            )
            rsa_public_key = []
            for r in results:
                if len(r[1]) == 2:
                    rsa_public_key = r[1]
                    self._client_manager.register_ce_server(r[0])
                
            #send RSA public key to all clients
            log(INFO, "Send RSA public key of the CE server to all clients")
            clients = self._client_manager.sample(min_num_clients,min_num_clients)
            client_instructions= [(client, rsa_public_key) for client in clients]
            results, failures = fn_clients(
                client_fn=send_public_key,
                client_instructions=client_instructions,
                max_workers=self.max_workers,
                timeout=timeout,
            )
        
        log(INFO, "Initializing global parameters")
        # Public Keys aggregation
        log(INFO, "#################SET PK#################")
        t0=time.time()
        pk_aggregated,self.n,self.clients = self.get_pk(timeout=timeout) #TODO fix timeout
        print(self.clients)
        self.client_mapping = {c.cid:i for i,c in enumerate(self.clients)}
        self.n_tot = self.n
        self.set_pk(pk_aggregated)
        self.send_pk(self.context,timeout=timeout)
        t01=time.time()-t0
        # Initialize parameters
        log(INFO, "#################GET INITIAL PARAMETERS#################")
        self.length = length
        t02=time.time()
        self.parameters = self.get_initial_parameters_enc(timeout=timeout)
        t03=time.time()-t02
        log(INFO, "Evaluating initial parameters")
        t04=time.time()
        res = self.evaluate_round_enc(0,history,timeout) #self.strategy.evaluate(0, parameters=self.parameters)
        t05=time.time()-t04
        if res is not None :
            c = {k: v for k, v in res[1].items() if k!="predictions"}
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                c,#res[1],
            )
        history.add_round_time(0, {"fit_round":t01+t03,"evaluate_round":t05})

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()
        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            log(INFO, "#################FIT ROUND#################")
            t=time.time()
            res_fit = self.fit_round_enc(
                server_round=current_round,
                current_round=current_round,
                start_time=start_time,
                history=history,
                timeout=timeout,
            )
            self.round_fit_time = time.time()-t
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )
            # Evaluate model on a sample of available clients
            t2=time.time()
            self.evaluate_round_enc(current_round,history,timeout)
            t3=time.time()-t2
            history.add_round_time(current_round, {"fit_round":self.round_fit_time,"evaluate_round":t3})
        self.eval_enc_last(num_rounds+1,start_time,history,timeout)    

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
    
    def change_n(self,n, clients,timeout):
        self.n = n
        self.clients = clients
        self.context = ts.context(ts.SCHEME_TYPE.MK_CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.context.global_scale = 2**40
        log(INFO, "Initializing global parameters")
        # Public Keys aggregation
        log(INFO, "#################SET PK#################")
        pk_aggregated, n, _ = self.get_pk(timeout=timeout) #TDO fix timeout
        if n != self.n:
            raise RuntimeError("Error while getting the public keys")
        self.set_pk(pk_aggregated)
        self.send_pk(self.context,timeout=timeout)
        # Initialize parameters
        log(INFO, "#################GET PARAMETERS#################")
        self.parameters = self.get_initial_parameters_enc(timeout=timeout)

    
    def eval_enc_last(
        self,
        server_round: int,
        start_time: float,
        history: History,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        clients = self.clients
        client_instructions = self.strategy.configure_fit_enc(
            (self.context,self.parameters) ,
            client_manager=self._client_manager, 
            clients = clients
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "eval_enc_last %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Send Encrypted parameters to all clients participating in this round
        # Collect `decryption shares` from all clients participating in this round
        results, failures = fn_clients(
            client_fn=send_enc,
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "eval_enc_last %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        if len(failures) != 0 or len(results) != self.n:
            raise RuntimeError("Error while getting the decryption shares")
        
        # Aggregate decryption shares
        aggregated_ds: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate(server_round, results, lambda x,y : x.add_share(y[1]),self.parameters,failures)
        
        parameters_aggregated, metrics_aggregated = aggregated_ds[0]*(1/self.n), aggregated_ds[1] #*(1/self.n)
        # Evaluate model using strategy implementation
        self.evaluate_enc(parameters_aggregated,server_round-1,start_time,history)

    
    def evaluate_enc(self,parameters,current_round, start_time, history):
        # Evaluate model using strategy implementation
        log(INFO, "#################EVALUATE MODEL ON GLOBAL SERVER#################")
        p=parameters.mk_decode() #TODO check division by n
        res_cen = self.strategy.evaluate_enc(current_round, parameters=p)#[x/self.n for x in p])
        if res_cen is not None:
            loss_cen, metrics_cen = res_cen
            c = {k: v for k, v in metrics_cen.items() if k!="predictions"}
            log(
                INFO,
                "Evaluation progress: (%s, %s, %s, %s)",
                current_round,
                loss_cen,
                c,#metrics_cen,
                timeit.default_timer() - start_time,
            )
            history.add_loss_centralized(server_round=current_round, loss=loss_cen)
            history.add_metrics_centralized(
                server_round=current_round, metrics=metrics_cen
            )

    def evaluate_round_enc(self,server_round,history,timeout):
        # Evaluate model on a sample of available clients
        log(INFO, "#################EVALUATE MODEL ON CLIENTS#################")

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate_enc(
            server_round=server_round,
            client_manager=self._client_manager,
            clients = self.clients
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = fn_clients(
            client_instructions,
            evaluate_client_enc,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )


        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        
        if aggregated_result is not None:
            loss_fed, evaluate_metrics_fed = aggregated_result
            if loss_fed is not None:
                history.add_loss_distributed(
                    server_round=server_round, loss=loss_fed
                )
                history.add_metrics_distributed(
                    server_round=server_round, metrics=evaluate_metrics_fed
                )
        return loss_aggregated, metrics_aggregated, (results, failures)

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters
    
    def example_request(self, client: ClientProxy) -> Tuple[str, int]:
        question = "Could you find the sum of the list, Bob?"
        l = [1, 2, 3]
        return client.request(question, l)
    


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res

def fn_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    client_fn,
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(client_fn, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fn(
            future=future, results=results, failures=failures
        )
    return results, failures


def get_pk_client(
    client: ClientProxy, ins, timeout: Optional[float]
) :# TODO-> Tuple[ClientProxy, ]:
    context, cid = ins
    """Refine parameters on a single client."""
    get_pk_res = client.get_pk(context,cid,timeout=timeout)
    pk=ts._ts_cpp.PublicKey(get_pk_res.data.ciphertext()[0])
    return client, get_pk_res

def send_client(
    client, fn, ins, timeout: Optional[float]
):
    """Refine parameters on a single client."""
    fit_res = fn(*ins, timeout=timeout)
    return client, fit_res

def send_pk_client(
    client: ClientProxy, ctx, timeout: Optional[float]
): #TODO -> Tuple[ClientProxy, SendPKRes]:
    """Refine parameters on a single client."""
    send_pk_res = client.send_pk(ctx, timeout=timeout)
    return client, send_pk_res

def get_parms_client(
    client: ClientProxy, ins, timeout: Optional[float]
) :# TODO-> Tuple[ClientProxy, ]:
    """Refine parameters on a single client."""
    cid = ins
    get_parms_res = client.get_parms(cid, timeout=timeout)
    return client, get_parms_res

def send_enc(
        client: ClientProxy, ins, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    
    context,enc,cid = ins
    fit_res = client.send_enc(context,enc,cid, timeout=timeout)
    return client, fit_res

def send_ds(
    client: ClientProxy, ins, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    #context,enc = ins
    #fit_res = client.send_ds(context,enc, timeout=timeout)
    fit_res = client.send_ds(None,ins, timeout=timeout)
    return client, fit_res

def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def _handle_finished_future_after_fn(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    results.append(result)
    return
    # Check result status code
    #if res.status.code == Code.OK:
    #    results.append(result)
    #    return
    #TODO check

    # Not successful, client returned a result where the status code is not OK
    #failures.append(result)


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client_enc, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fn(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res

def evaluate_client_enc(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate_enc(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
    
    
def get_gradients(
    client: ClientProxy, ins, timeout: Optional[float]
) :
    get_gradients_res = client.get_gradients(timeout=timeout)
    return client, get_gradients_res
    
def get_contributions(
    client: ClientProxy, ins, timeout: Optional[float]
) :
    get_contribution_res = client.get_contributions(ins,timeout=timeout)
    return client, get_contribution_res
    
def identify(
    client: ClientProxy, ins, timeout: Optional[float]
) :
    status = client.identify(timeout=timeout)
    return client, status
    
def send_public_key(
    client: ClientProxy, ins, timeout: Optional[float]
) :
    status = client.send_public_key(ins,timeout=timeout)
    return client, status
    
    
