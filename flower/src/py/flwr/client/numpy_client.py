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
"""Flower client app."""


from abc import ABC
from typing import Callable, Dict, Tuple, List

from flwr.client.client import Client
from flwr.client.run_state import RunState
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    serde,
)
from flwr.common.typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
)
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT = """
NumPyClient.fit did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[NDArrays, int, Dict[str, Scalar]]

Example
-------

    model.get_weights(), 10, {"accuracy": 0.95}

"""

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE = """
NumPyClient.evaluate did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[float, int, Dict[str, Scalar]]

Example
-------

    0.5, 10, {"accuracy": 0.95}

"""


class NumPyClient(ABC):
    """Abstract base class for Flower clients using NumPy."""

    state: RunState

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Return a client's set of properties.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which properties
            are needed along with some Scalar attributes.

        Returns
        -------
        properties : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary property values back to the server.
        """
        _ = (self, config)
        return {}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which parameters
            are needed along with some Scalar attributes.

        Returns
        -------
        parameters : NDArrays
            The local model parameters as a list of NumPy ndarrays.
        """
        _ = (self, config)
        return []

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : NDArrays
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """
        _ = (self, parameters, config)
        return [], 0, {}
    
    def fit_enc(
        self, parameters: NDArrays, config: Dict[str, Scalar], flat=False
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : NDArrays
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """
        _ = (self, parameters, config)
        return [], 0, {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) have been
        deprecated and removed since Flower 0.19.
        """
        _ = (self, parameters, config)
        return 0.0, 0, {}
    
    def evaluate_enc(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) have been
        deprecated and removed since Flower 0.19.
        """
        _ = (self, parameters, config)
        return 0.0, 0, {}

    def get_state(self) -> RunState:
        """Get the run state from this client."""
        return self.state

    def set_state(self, state: RunState) -> None:
        """Apply a run state to this client."""
        self.state = state

    def to_client(self) -> Client:
        """Convert to object to Client type and return it."""
        return _wrap_numpy_client(client=self)
    
    def requestSum(self, status: str, val: List[int]) -> Tuple[str, int]:
        """Request sum of list of val from server."""
        request_msg = serde.val_to_proto(status,val)
        response_msg : ServerMessage = self.bridge.request(
            ClientMessage(val=request_msg)
        )
        status, sum = serde.sum_from_proto(response_msg.sum)
        return status, sum
    
    def example_response(self, question: str, l: List[int]) -> Tuple[str, int]:
        response = "Here you go Alice!"
        answer = sum(l)
        return response, answer
        
    def identify(self):
        return [""]
        
    def set_public_key(self,publickey):
        return


def has_get_properties(client: NumPyClient) -> bool:
    """Check if NumPyClient implements get_properties."""
    return type(client).get_properties != NumPyClient.get_properties


def has_get_parameters(client: NumPyClient) -> bool:
    """Check if NumPyClient implements get_parameters."""
    return type(client).get_parameters != NumPyClient.get_parameters

def has_fit(client: NumPyClient) -> bool:
    """Check if NumPyClient implements fit."""
    return type(client).fit != NumPyClient.fit

def has_fit_enc(client: NumPyClient) -> bool:
    """Check if NumPyClient implements fit."""
    return type(client).fit_enc != NumPyClient.fit_enc


def has_evaluate(client: NumPyClient) -> bool:
    """Check if NumPyClient implements evaluate."""
    return type(client).evaluate != NumPyClient.evaluate

def has_evaluate_enc(client: NumPyClient) -> bool:
    """Check if NumPyClient implements evaluate_enc."""
    return type(client).evaluate_enc != NumPyClient.evaluate_enc
    
def has_identify(client: NumPyClient) -> bool:
    """Check if NumPyClient implements evaluate_enc."""
    return callable(getattr(client, "identify", None))
    
def has_set_public_key(client: NumPyClient) -> bool:
    """Check if NumPyClient implements evaluate_enc."""
    return callable(getattr(client, "set_public_key", None))

def has_example_response(client: NumPyClient) -> bool:
    return callable(getattr(client, "example_response", None))


def _constructor(self: Client, numpy_client: NumPyClient) -> None:
    self.numpy_client = numpy_client  # type: ignore


def _get_properties(self: Client, ins: GetPropertiesIns) -> GetPropertiesRes:
    """Return the current client properties."""
    properties = self.numpy_client.get_properties(config=ins.config)  # type: ignore
    return GetPropertiesRes(
        status=Status(code=Code.OK, message="Success"),
        properties=properties,
    )


def _get_parameters(self: Client, ins: GetParametersIns) -> GetParametersRes:
    """Return the current local model parameters."""
    parameters = self.numpy_client.get_parameters(config=ins.config)  # type: ignore
    parameters_proto = ndarrays_to_parameters(parameters)
    return GetParametersRes(
        status=Status(code=Code.OK, message="Success"), parameters=parameters_proto
    )


def _fit(self: Client, ins: FitIns) -> FitRes:
    """Refine the provided parameters using the locally held dataset."""
    # Deconstruct FitIns
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    # Train
    results = self.numpy_client.fit(parameters, ins.config)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], list)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT)

    # Return FitRes
    parameters_prime, num_examples, metrics = results
    parameters_prime_proto = ndarrays_to_parameters(parameters_prime)
    return FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=parameters_prime_proto,
        num_examples=num_examples,
        metrics=metrics,
    )

def _fit_enc(self: Client, ins: FitIns,config=None,flat=False) -> FitRes:
    """Refine the provided parameters using the locally held dataset."""
    # Deconstruct FitIns
    #parameters: NDArrays = _fitparameters_to_ndarrays(ins.parameters)

    # Train
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)
    results = self.numpy_client.fit_enc(parameters, config,flat=flat)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], list)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT)

    # Return FitRes
    parameters_prime, num_examples, metrics = results
    #parameters_prime_proto = ndarrays_to_parameters(parameters_prime)
    #return FitRes(
    #    status=Status(code=Code.OK, message="Success"),
    #    parameters=parameters_prime_proto,
    #    num_examples=num_examples,
    #    metrics=metrics,
    #)
    return parameters_prime ,num_examples, metrics

def _evaluate(self: Client, ins: EvaluateIns) -> EvaluateRes:
    """Evaluate the provided parameters using the locally held dataset."""
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    results = self.numpy_client.evaluate(parameters, ins.config)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], float)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE)

    # Return EvaluateRes
    loss, num_examples, metrics = results
    return EvaluateRes(
        status=Status(code=Code.OK, message="Success"),
        loss=loss,
        num_examples=num_examples,
        metrics=metrics,
    )

def _evaluate_enc(self: Client, ins: EvaluateIns, reshape=False) -> EvaluateRes:
    """Evaluate the provided parameters using the locally held dataset."""
    #parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    results = self.numpy_client.evaluate_enc(ins,reshape)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], float)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE)

    # Return EvaluateRes
    loss, num_examples, metrics = results
    #return EvaluateRes(
    #    status=Status(code=Code.OK, message="Success"),
    #    loss=loss,
    #    num_examples=num_examples,
    #    metrics=metrics,
    #)
    return loss, num_examples, metrics


def _get_state(self: Client) -> RunState:
    """Return state of underlying NumPyClient."""
    return self.numpy_client.get_state()  # type: ignore


def _set_state(self: Client, state: RunState) -> None:
    """Apply state to underlying NumPyClient."""
    self.numpy_client.set_state(state)  # type: ignore

def _identify(self):
    return self.numpy_client.identify()
    
def _set_public_key(self, public_key):
    return self.numpy_client.set_public_key(public_key)

def _wrap_numpy_client(client: NumPyClient) -> Client:
    member_dict: Dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
        "get_state": _get_state,
        "set_state": _set_state,
    }

    # Add wrapper type methods (if overridden)

    if has_get_properties(client=client):
        member_dict["get_properties"] = _get_properties

    if has_get_parameters(client=client):
        member_dict["get_parameters"] = _get_parameters

    if has_fit(client=client):
        member_dict["fit"] = _fit
    
    if has_fit_enc(client=client):
        member_dict["fit_enc"] = _fit_enc

    if has_evaluate(client=client):
        member_dict["evaluate"] = _evaluate
    
    if has_evaluate_enc(client=client):
        member_dict["evaluate_enc"] = _evaluate_enc

    if has_example_response(client=client):
        member_dict["example_response"] = _example_response
        
    if has_identify(client=client):
        member_dict["identify"] = _identify
    
    if has_set_public_key(client=client):
        member_dict["set_public_key"] = _set_public_key

    # Create wrapper class
    wrapper_class = type("NumPyClientWrapper", (Client,), member_dict)

    # Create and return an instance of the newly created class
    return wrapper_class(numpy_client=client)  # type: ignore

def _example_response(self, question: str, l: List[int]) -> Tuple[str, int]:
    return self.numpy_client.example_response(question, l)
