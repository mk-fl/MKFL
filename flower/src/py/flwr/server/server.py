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
import random

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
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
import time
        
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
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None
        self.pk = None
        self.clients = []
        self.client_mapping = None
        self.n = 0
        self.ce=contribution
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

    def identify_ce_server(self, timeout: Optional[float]):
        log(INFO, "Identify the CE server")
        min_num_clients = self.strategy.min_available_clients
        clients = self._client_manager.sample(min_num_clients+1,min_num_clients+1,timeout = 300)
        if clients == []:
            return -1
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
            else:
                self.clients.append(r[0])
                
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
        self.client_mapping = {c.cid:i for i,c in enumerate(self.clients)}
        return 0
                
    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float],length) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Identify ce server
        if self.ce:
            if self.identify_ce_server(timeout=timeout):
                return history
        
        # Initialize parameters
        log(INFO, "Initializing global parameters")
        t0=time.time()
        self.parameters = self._get_initial_parameters(timeout=timeout)
        t01=time.time()-t0
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        t1=time.time()
        res_fed = self.evaluate_round(server_round=0, timeout=timeout)
        t11=time.time()-t1
        history.add_round_time(0, {"fit_round": t01, "evaluate_round": t11})
        if res_fed is not None:
            loss_fed, evaluate_metrics_fed, _ = res_fed
            if loss_fed is not None:
                history.add_loss_distributed(
                    server_round=0, loss=loss_fed
                )
                history.add_metrics_distributed(
                    server_round=0, metrics=evaluate_metrics_fed
                )
        self.n = self.strategy.min_available_clients
        if res is not None:
            c = {k: v for k, v in res[1].items() if k!="predictions"}
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                c,#res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        self.example_request(self._client_manager.sample(1)[0])
        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()
        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            print("################### FIT ROUND ##########################")
            t2=time.time()
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            t3=time.time()-t2
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            print("################### EVALUATE ON GLOBAL SERVER ##########################")
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                c = {k: v for k, v in metrics_cen.items() if k!="predictions"}
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    c,#metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            print("################### EVALUATE ON CLIENTS ##########################")
            t4=time.time()
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            t5=time.time()-t4
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
            history.add_round_time(current_round, {"fit_round": t3, "evaluate_round": t5})

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

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
    
    def shapley_round(self, server_round: int, timeout: Optional[float]):
        # send global parameters to the ce server
        if self.check:
            instruction = EvaluateIns(parameters=self.parameters,config={})
            evaluate_client(self._client_manager.ce_server, instruction, timeout)
            self.check = False
        # compute the reputation of each client
        self.check = not random.randrange(5)
        changed = False
        if self.check:
            shapley_values = self.compute_reputation(server_round, timeout)
            log(INFO, "Shapley values round " + str(server_round) + " : " + str([(self.client_mapping[x.cid], shapley_values[x]) for x in shapley_values]))
            # eliminate a client with a low shapley value
            changed = self.eliminate_clients(shapley_values,server_round, timeout)
        return changed
            
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
        
        #ici
        if self.methodo == "set_aside":
            fit_ins = FitIns(Parameters(tensors=[], tensor_type="numpy.ndarray"), {"wait":"no_update"})
            client_instructions += [(client, fit_ins) for client in self.waiting]
            log(INFO, "set_aside: " + str(len(self.clients)) + " active clients and " + str(len(self.waiting)) + " waiting clients")
        elif self.methodo == "set_aside2":
            fit_ins = FitIns(self.parameters, {"wait":"global_update"})
            client_instructions += [(client, fit_ins) for client in self.waiting]
            log(INFO, "set_aside2: " + str(len(self.clients)) + " active clients and " + str(len(self.waiting)) + " waiting clients")
        
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
        if self.ce:
            changed = self.shapley_round(server_round, timeout)
            results = [x for x in results if x[0] in self.clients]
        # Aggregate training results
        print("################### AGGREGATE FIT ##########################")
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit2(self.n,server_round, results, failures)
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


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
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
        _handle_finished_future_after_evaluate(
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
    
