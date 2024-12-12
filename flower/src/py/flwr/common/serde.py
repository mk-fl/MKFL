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
"""ProtoBuf serialization and deserialization."""

from flwr.common.logger import log
from logging import INFO, WARN
from typing import Any, Dict, List, MutableMapping, cast, Tuple

from flwr.proto.task_pb2 import Value
from flwr.proto.transport_pb2 import (
    ClientMessage,
    Code,
    Parameters,
    Reason,
    Scalar,
    ServerMessage,
    Status,
)
from flwr.common import (
    NDArrays,
    ndarray_to_bytes,
    bytes_to_ndarray,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from . import typing

import sys
sys.path.append('../../../../../TenSEAL')
import tenseal as ts

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

#  === ServerMessage message ===


def server_message_to_proto(server_message: typing.ServerMessage) -> ServerMessage:
    """Serialize `ServerMessage` to ProtoBuf."""
    if server_message.get_properties_ins is not None:
        return ServerMessage(
            get_properties_ins=get_properties_ins_to_proto(
                server_message.get_properties_ins,
            )
        )
    if server_message.get_parameters_ins is not None:
        return ServerMessage(
            get_parameters_ins=get_parameters_ins_to_proto(
                server_message.get_parameters_ins,
            )
        )
    if server_message.fit_ins is not None:
        return ServerMessage(
            fit_ins=fit_ins_to_proto(
                server_message.fit_ins,
            )
        )
    if server_message.evaluate_ins is not None:
        return ServerMessage(
            evaluate_ins=evaluate_ins_to_proto(
                server_message.evaluate_ins,
            )
        )
    if server_message.get_pk_ins is not None:
        return ServerMessage(
            get_pk_ins=server_message.get_pk_ins_to_proto()
        )
    raise Exception("No instruction set in ServerMessage, cannot serialize to ProtoBuf")


def server_message_from_proto(
    server_message_proto: ServerMessage,
) -> typing.ServerMessage:
    """Deserialize `ServerMessage` from ProtoBuf."""
    field = server_message_proto.WhichOneof("msg")
    if field == "get_properties_ins":
        return typing.ServerMessage(
            get_properties_ins=get_properties_ins_from_proto(
                server_message_proto.get_properties_ins,
            )
        )
    if field == "get_parameters_ins":
        return typing.ServerMessage(
            get_parameters_ins=get_parameters_ins_from_proto(
                server_message_proto.get_parameters_ins,
            )
        )
    if field == "fit_ins":
        return typing.ServerMessage(
            fit_ins=fit_ins_from_proto(
                server_message_proto.fit_ins,
            )
        )
    if field == "evaluate_ins":
        return typing.ServerMessage(
            evaluate_ins=evaluate_ins_from_proto(
                server_message_proto.evaluate_ins,
            )
        )
    raise Exception(
        "Unsupported instruction in ServerMessage, cannot deserialize from ProtoBuf"
    )

def sum_to_proto(status: str, sum: List[int]) -> ServerMessage.SendSumIns:
    """Serialize `SendSum` to ProtoBuf."""
    return ServerMessage.SendSumIns(status=status, sum=sum)

def sum_from_proto(msg: ServerMessage.SendSumIns):
    """Deserialize `SendSum` from ProtoBuf."""
    return msg.status, msg.sum

def example_msg_to_proto(question: str, l: List[int]) -> ServerMessage.SendSumIns:
    return ServerMessage.SendSumIns(status=question, sum=l)


def example_msg_from_proto(msg: ServerMessage.SendSumIns) -> Tuple[str, List[int]]:
    return msg.status, msg.sum


def example_res_to_proto(response: str, answer: int) -> ClientMessage.SendValRes:
    print(f"response: {type(response)}, answer: {type(answer)}")
    return ClientMessage.SendValRes(status=response, val=answer)


def example_res_from_proto(res: ClientMessage.SendValRes) -> Tuple[str, int]:
    return res.status, res.val

#  === ClientMessage message ===


def client_message_to_proto(client_message: typing.ClientMessage) -> ClientMessage:
    """Serialize `ClientMessage` to ProtoBuf."""
    if client_message.get_properties_res is not None:
        return ClientMessage(
            get_properties_res=get_properties_res_to_proto(
                client_message.get_properties_res,
            )
        )
    if client_message.get_parameters_res is not None:
        return ClientMessage(
            get_parameters_res=get_parameters_res_to_proto(
                client_message.get_parameters_res,
            )
        )
    if client_message.fit_res is not None:
        return ClientMessage(
            fit_res=fit_res_to_proto(
                client_message.fit_res,
            )
        )
    if client_message.evaluate_res is not None:
        return ClientMessage(
            evaluate_res=evaluate_res_to_proto(
                client_message.evaluate_res,
            )
        )
    raise Exception("No instruction set in ClientMessage, cannot serialize to ProtoBuf")


def client_message_from_proto(
    client_message_proto: ClientMessage,
) -> typing.ClientMessage:
    """Deserialize `ClientMessage` from ProtoBuf."""
    field = client_message_proto.WhichOneof("msg")
    if field == "get_properties_res":
        return typing.ClientMessage(
            get_properties_res=get_properties_res_from_proto(
                client_message_proto.get_properties_res,
            )
        )
    if field == "get_parameters_res":
        return typing.ClientMessage(
            get_parameters_res=get_parameters_res_from_proto(
                client_message_proto.get_parameters_res,
            )
        )
    if field == "fit_res":
        return typing.ClientMessage(
            fit_res=fit_res_from_proto(
                client_message_proto.fit_res,
            )
        )
    if field == "evaluate_res":
        return typing.ClientMessage(
            evaluate_res=evaluate_res_from_proto(
                client_message_proto.evaluate_res,
            )
        )
    raise Exception(
        "Unsupported instruction in ClientMessage, cannot deserialize from ProtoBuf"
    )

def val_to_proto(status: str, val: List[int]) -> ClientMessage.SendValRes:
    """Serialize `SendVal` to ProtoBuf."""
    return ServerMessage.SendValRes(status=status, val=val)

def val_from_proto(msg:ClientMessage.SendValRes):
    """Deserialize `SendValRes` from ProtoBuf."""
    return msg.status, msg.val


#  === Parameters message ===


def parameters_to_proto(parameters: typing.Parameters) -> Parameters:
    """Serialize `Parameters` to ProtoBuf."""
    p=Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)
    l=0
    for t in parameters.tensors:
        l+=sys.getsizeof(t)
    #log(INFO,f"parameters_to_proto: {get_size(p)} {sys.getsizeof(p)} {l+sys.getsizeof(parameters.tensor_type)} {type(parameters.tensors[0])} {type(parameters.tensor_type)}")
    #return p
    return Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)


def parameters_from_proto(msg: Parameters) -> typing.Parameters:
    """Deserialize `Parameters` from ProtoBuf."""
    tensors: List[bytes] = list(msg.tensors)
    return typing.Parameters(tensors=tensors, tensor_type=msg.tensor_type)


#  === ReconnectIns message ===


def reconnect_ins_to_proto(ins: typing.ReconnectIns) -> ServerMessage.ReconnectIns:
    """Serialize `ReconnectIns` to ProtoBuf."""
    if ins.seconds is not None:
        return ServerMessage.ReconnectIns(seconds=ins.seconds)
    return ServerMessage.ReconnectIns()


def reconnect_ins_from_proto(msg: ServerMessage.ReconnectIns) -> typing.ReconnectIns:
    """Deserialize `ReconnectIns` from ProtoBuf."""
    return typing.ReconnectIns(seconds=msg.seconds)


# === DisconnectRes message ===


def disconnect_res_to_proto(res: typing.DisconnectRes) -> ClientMessage.DisconnectRes:
    """Serialize `DisconnectRes` to ProtoBuf."""
    reason_proto = Reason.UNKNOWN
    if res.reason == "RECONNECT":
        reason_proto = Reason.RECONNECT
    elif res.reason == "POWER_DISCONNECTED":
        reason_proto = Reason.POWER_DISCONNECTED
    elif res.reason == "WIFI_UNAVAILABLE":
        reason_proto = Reason.WIFI_UNAVAILABLE
    return ClientMessage.DisconnectRes(reason=reason_proto)


def disconnect_res_from_proto(msg: ClientMessage.DisconnectRes) -> typing.DisconnectRes:
    """Deserialize `DisconnectRes` from ProtoBuf."""
    if msg.reason == Reason.RECONNECT:
        return typing.DisconnectRes(reason="RECONNECT")
    if msg.reason == Reason.POWER_DISCONNECTED:
        return typing.DisconnectRes(reason="POWER_DISCONNECTED")
    if msg.reason == Reason.WIFI_UNAVAILABLE:
        return typing.DisconnectRes(reason="WIFI_UNAVAILABLE")
    return typing.DisconnectRes(reason="UNKNOWN")


# === GetParameters messages ===


def get_parameters_ins_to_proto(
    ins: typing.GetParametersIns,
) -> ServerMessage.GetParametersIns:
    """Serialize `GetParametersIns` to ProtoBuf."""
    config = properties_to_proto(ins.config)
    #return ServerMessage.GetParametersIns(config=config)
    p=ServerMessage.GetParametersIns(config=config)
    #log(INFO,f"get_parameters_ins_to_proto: {get_size(p)} {sys.getsizeof(p)} {sys.getsizeof(config)}")
    return p


def get_parameters_ins_from_proto(
    msg: ServerMessage.GetParametersIns,
) -> typing.GetParametersIns:
    """Deserialize `GetParametersIns` from ProtoBuf."""
    config = properties_from_proto(msg.config)
    return typing.GetParametersIns(config=config)


def get_parameters_res_to_proto(
    res: typing.GetParametersRes,
) -> ClientMessage.GetParametersRes:
    """Serialize `GetParametersRes` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        return ClientMessage.GetParametersRes(status=status_msg)
    parameters_proto = parameters_to_proto(res.parameters)
    #return ClientMessage.GetParametersRes(
    #    status=status_msg, parameters=parameters_proto
    #)
    p = ClientMessage.GetParametersRes(
        status=status_msg, parameters=parameters_proto
    )
    #log(INFO,f"get_parameters_res_to_proto: {get_size(p)} {sys.getsizeof(p)} {sys.getsizeof(parameters_proto)+sys.getsizeof(status_msg)}")
    return p


def get_parameters_res_from_proto(
    msg: ClientMessage.GetParametersRes,
) -> typing.GetParametersRes:
    """Deserialize `GetParametersRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    parameters = parameters_from_proto(msg.parameters)
    return typing.GetParametersRes(status=status, parameters=parameters)


# === GetContributions messages ===


def get_contributions_ins_to_proto(
    ins
) -> ServerMessage.GetContributionsIns:
    """Serialize `GetContributionsIns` to ProtoBuf."""
    return ServerMessage.GetContributionsIns(gradients=ins)

def get_contributions_ins_from_proto(
    msg: ServerMessage.GetContributionsIns
):
    """Deserialize `GetContributionsIns` from ProtoBuf."""
    return msg.gradients

def get_contributions_res_to_proto(
    res
) -> ClientMessage.GetContributionsRes:
    """Serialize `GetContributionsRes` to ProtoBuf."""
    return ClientMessage.GetContributionsRes(contributions=res)

def get_contributions_res_from_proto(
    msg: ClientMessage.GetContributionsRes
):
    """Deserialize `GetContributionsRes` from ProtoBuf."""
    return msg.contributions
    
    
# === GetGradients messages ===


def get_gradients_ins_to_proto():
    return ServerMessage.GetGradientsIns()

def get_gradients_ins_from_proto(msg:ServerMessage.GetGradientsIns):    
    return 

def get_gradients_res_to_proto(
    res,
) -> ClientMessage.GetGradientsRes:
    """Serialize `GetGradientsRes` to ProtoBuf."""
    return ClientMessage.GetGradientsRes(gradients=res)

def get_gradients_res_from_proto(
    msg: ClientMessage.GetGradientsRes,
):
    """Deserialize `GetGradientsRes` from ProtoBuf."""
    return msg.gradients

# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> ServerMessage.FitIns:
    """Serialize `FitIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    #return ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)
    p = ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)
    #log(INFO,f"fit_ins_to_proto: {get_size(p)} {sys.getsizeof(p)}  {sys.getsizeof(parameters_proto)+sys.getsizeof(config_msg)}")
    return p


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize `FitIns` from ProtoBuf."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    """Serialize `FitIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.FIT_NOT_IMPLEMENTED:
        return ClientMessage.FitRes(status=status_msg)
    parameters_proto = parameters_to_proto(res.parameters)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    #return ClientMessage.FitRes(
    #    status=status_msg,
    #    parameters=parameters_proto,
    #    num_examples=res.num_examples,
    #    metrics=metrics_msg,
    #)
    p= ClientMessage.FitRes(
        status=status_msg,
        parameters=parameters_proto,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )
    #log(INFO,f"fit_res_to_proto: {get_size(p)} {sys.getsizeof(p)}   {sys.getsizeof(parameters_proto)+sys.getsizeof(metrics_msg)+sys.getsizeof(status_msg)}")
    return p


def fit_res_from_proto(msg: ClientMessage.FitRes) -> typing.FitRes:
    """Deserialize `FitRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    parameters = parameters_from_proto(msg.parameters)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.FitRes(
        status=status,
        parameters=parameters,
        num_examples=msg.num_examples,
        metrics=metrics,
    )

# === GetPK/SendPK messages ===

ctx_g = None

#send server's pk
def get_pk_ins_to_proto(ctx) -> ServerMessage.GetPKIns:
    """Serialize `GetPKIns` to ProtoBuf."""
    ctx_proto = ctx.serialize()
    #return ServerMessage.GetPKIns(ctx = ctx_proto)
    p= ServerMessage.GetPKIns(ctx = ctx_proto)
    #log(INFO,f"get_pk_ins_to_proto: {get_size(p)} {sys.getsizeof(p)} {sys.getsizeof(ctx_proto)}")
    return p


def get_pk_ins_from_proto(msg: ServerMessage.GetPKIns) -> typing.GetPKIns:
    """Deserialize `GetPKIns` from ProtoBuf."""
    ctx = ts.context_from(msg.ctx)
    return  ctx.public_key() 

#send each individual pk
def get_pk_res_to_proto(ctx) -> ClientMessage.GetPKRes:
    """Serialize `GetPKRes` to ProtoBuf."""
    ctx_proto = ctx.serialize()
    pk = ctx.public_key()
    pk_ckks_proto = ts.ckks_vector(ctx,pk).serialize()
    #return ClientMessage.GetPKRes(ctx=ctx_proto)
    p=ClientMessage.GetPKRes(ctx=ctx_proto)
    #log(INFO,f"get_pk_res_to_proto: {get_size(p)} {sys.getsizeof(p)} {sys.getsizeof(ctx_proto)}")
    return p


def get_pk_res_from_proto(msg: ClientMessage.GetPKRes) :
    """Deserialize `GetPKRes` from ProtoBuf."""
    ctx = ts.context_from(msg.ctx)
    pk = ts.ckks_vector(ctx, ctx.public_key())
    global ctx_g
    ctx_g=ctx
    return pk

#send aggregated pk
def send_pk_ins_to_proto(ctx) -> ServerMessage.SendPKIns:
    """Serialize `SendPKIns` to ProtoBuf."""
    ctx_proto = ctx.serialize()
    pk = ctx.public_key()
    pk_ckks_proto = ts.ckks_vector(ctx,pk).serialize()
    #return ServerMessage.SendPKIns(ctx=ctx_proto)
    p=ServerMessage.SendPKIns(ctx=ctx_proto)
    #log(INFO,f"send_pk_ins_to_proto: {get_size(p)} {sys.getsizeof(p)} {sys.getsizeof(ctx_proto)}")
    return p


def send_pk_ins_from_proto(msg: ServerMessage.SendPKIns) -> typing.SendPKIns:
    """Deserialize `SendPKIns` from ProtoBuf."""
    ctx = ts.context_from(msg.ctx)
    pk = ts.ckks_vector(ctx,ctx.public_key())
    global ctx_g
    ctx_g=ctx
    return pk

#send "ok" or "ko"
def send_pk_res_to_proto(status) -> ClientMessage.SendPKRes:
    """Serialize `SendPKRes` to ProtoBuf."""
    #return ClientMessage.SendPKRes(status=status)
    p= ClientMessage.SendPKRes(status=status)
    #log(INFO,f"send_pk_res_to_proto: {get_size(p)} {sys.getsizeof(p)}   {sys.getsizeof(status)}")
    return p


def send_pk_res_from_proto(msg: ClientMessage.SendPKRes) -> typing.SendPKRes:
    """Deserialize `SendPKRes` from ProtoBuf."""
    return msg.status

# === GetParms messages ===
#send request
def get_parms_ins_to_proto() -> ServerMessage.GetParmsIns:
    """Serialize `GetParmsIns` to ProtoBuf."""
    #return ServerMessage.GetParmsIns()
    p=ServerMessage.GetParmsIns()
    #log(INFO,f"get_parms_ins_to_proto: {get_size(p)} {sys.getsizeof(p)}")
    return p

def get_parms_ins_from_proto(msg: ServerMessage.GetParmsIns):    
    return 

#send initial parameters
def get_parms_res_to_proto(ctx,parms) -> ClientMessage.GetParmsRes:
    """Serialize `GetParmsRes` to ProtoBuf."""
    ctx_proto = ctx.serialize()
    parms_proto = parms.serialize()
    #return ClientMessage.GetParmsRes(ctx=ctx_proto, parms=parms_proto)
    p= ClientMessage.GetParmsRes(ctx=ctx_proto, parms=parms_proto)
    #log(INFO,f"get_parms_res_to_proto: {get_size(p) } {sys.getsizeof(p)} {sys.getsizeof(parms_proto)+sys.getsizeof(ctx_proto)} {sys.getsizeof(ctx_proto)} {sys.getsizeof(parms_proto)}")
    return p

def get_parms_res_from_proto(msg: ClientMessage.GetParmsRes) :        
    """Deserialize `GetParmsRes` from ProtoBuf."""
    ctx = ts.context_from(msg.ctx)
    #ctx=ctx_g
    parms = ts.ckks_vector_from(ctx, msg.parms)
    return parms


# === SendEnc messages ===
#send encrypted vector
def send_enc_ins_to_proto(ctx,enc):
    """Serialize `SendEncIns` to ProtoBuf."""
    ctx_proto = ctx.serialize()
    enc_proto = enc.serialize()
    #return ServerMessage.SendEncIns(ctx=ctx_proto, enc = enc_proto)
    p= ServerMessage.SendEncIns(ctx=ctx_proto, enc = enc_proto)
    #log(INFO,f"send_enc_ins_to_proto: {get_size(p)} {sys.getsizeof(p)} {sys.getsizeof(ctx_proto)+sys.getsizeof(enc_proto)} {sys.getsizeof(ctx_proto)} {sys.getsizeof(enc_proto)}")
    return p

def send_enc_ins_from_proto(msg : ServerMessage.SendEncIns):
    """Deserialize `SendEncIns` from ProtoBuf."""
    ctx = ts.context_from(msg.ctx)
    #ctx=ctx_g
    enc = ts.ckks_vector_from(ctx, msg.enc)
    return enc

#send each individual decryption share
def send_enc_res_to_proto(ctx,ds):
    """Serialize `SendEncRes` to ProtoBuf."""
    ctx_proto = ctx.serialize()
    ds_proto = ts.PlaintextVector(ds).serialize()
    #return ClientMessage.SendEncRes(ctx=ctx_proto, ds = ds_proto)
    p= ClientMessage.SendEncRes(ctx=ctx_proto, ds = ds_proto)
    #log(INFO,f"send_enc_res_to_proto: {get_size(p)} {sys.getsizeof(p)} {sys.getsizeof(ctx_proto)+sys.getsizeof(ds_proto)} {sys.getsizeof(ctx_proto)} {sys.getsizeof(ds_proto)}")
    return p

def send_enc_res_from_proto(msg: ClientMessage.SendEncRes):
    """Deserialize `SendEncRes` from ProtoBuf."""
    #ctx = ts.context_from(msg.ctx)
    ctx=ctx_g
    ds = ts.plaintext_vector_from(ctx,msg.ds)
    return ds.data.plaintext()

# === SendDS messages ===
#send aggregated decryption share
def send_ds_ins_to_proto(ctx, ins):
    """Serialize `SendDSIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto({})
    #return ServerMessage.SendDSIns(parameters=parameters_proto, config=config_msg)
    p = ServerMessage.SendDSIns(parameters=parameters_proto, config=config_msg)
    #log(INFO,f"send_ds_ins_to_proto: {get_size(p)} {sys.getsizeof(p)}  {sys.getsizeof(parameters_proto)+sys.getsizeof(config_msg)}   {sys.getsizeof(parameters_proto)} {sys.getsizeof(config_msg)}")
   
    return p

def send_ds_ins_from_proto(msg:ServerMessage.SendDSIns):
    """Deserialize `SendDSIns` from ProtoBuf."""
    #ctx = ts.context_from(msg.ctx)
    #ds = ts.ckks_vector_from(ctx, msg.ds).mk_decode()
    #return ds 
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)

#send new parameters
def send_ds_res_to_proto(ctx, enc, loss=0, num_example=0, metrics={}):
    """Serialize `SendDSRes` to ProtoBuf."""
    ctx_proto = ctx.serialize()
    enc_proto = enc.serialize()
    metrics_msg = None if metrics is None else metrics_to_proto(metrics)
    #return ClientMessage.SendDSRes(ctx = ctx_proto, enc = enc_proto, loss=loss, num_examples=num_example, metrics = metrics_msg)
    p= ClientMessage.SendDSRes(ctx = ctx_proto, enc = enc_proto, loss=loss, num_examples=num_example, metrics = metrics_msg)
    #log(INFO,f"send_ds_res_to_proto: {get_size(p)} {sys.getsizeof(p)} {sys.getsizeof(ctx_proto)+sys.getsizeof(enc_proto)+sys.getsizeof(metrics_msg)+sys.getsizeof(loss)+sys.getsizeof(num_example)} {sys.getsizeof(ctx_proto)} {sys.getsizeof(enc_proto)}")
    return p
def send_ds_res_from_proto(msg:ClientMessage.SendDSRes):
    """Deserialize `SendDSRes` from ProtoBuf."""
    ctx = ts.context_from(msg.ctx)
    #ctx=ctx_g
    enc = ts.ckks_vector_from(ctx, msg.enc)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return enc, typing.EvaluateRes(
        status=None,
        loss=msg.loss,
        num_examples=msg.num_examples,
        metrics=metrics,
    )

# === SendEval messages ===
#send evaluation request
def send_eval_ins_to_proto():
    """Serialize `SendEvalIns` to ProtoBuf."""
    return ServerMessage.SendEvalIns()

def send_eval_ins_from_proto(msg:ServerMessage.SendEvalIns):    
    return 

#send evaluation results
def send_eval_res_to_proto(acc,n, loss):
    """Serialize `SendEvalRes` to ProtoBuf."""
    metrics_msg = None if acc is None else metrics_to_proto(acc)
    return ClientMessage.SendEvalRes(loss=loss, num_examples= n, metrics = metrics_msg)

def send_eval_res_from_proto(msg:ClientMessage.SendEvalRes):
    """Deserialize `SendEvalRes` from ProtoBuf."""
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        status=None,
        loss=msg.loss,
        num_examples=msg.num_examples,
        metrics=metrics,
    )

# === GetProperties messages ===


def get_properties_ins_to_proto(
    ins: typing.GetPropertiesIns,
) -> ServerMessage.GetPropertiesIns:
    """Serialize `GetPropertiesIns` to ProtoBuf."""
    config = properties_to_proto(ins.config)
    return ServerMessage.GetPropertiesIns(config=config)


def get_properties_ins_from_proto(
    msg: ServerMessage.GetPropertiesIns,
) -> typing.GetPropertiesIns:
    """Deserialize `GetPropertiesIns` from ProtoBuf."""
    config = properties_from_proto(msg.config)
    return typing.GetPropertiesIns(config=config)


def get_properties_res_to_proto(
    res: typing.GetPropertiesRes,
) -> ClientMessage.GetPropertiesRes:
    """Serialize `GetPropertiesIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        return ClientMessage.GetPropertiesRes(status=status_msg)
    properties_msg = properties_to_proto(res.properties)
    return ClientMessage.GetPropertiesRes(status=status_msg, properties=properties_msg)


def get_properties_res_from_proto(
    msg: ClientMessage.GetPropertiesRes,
) -> typing.GetPropertiesRes:
    """Deserialize `GetPropertiesRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    properties = properties_from_proto(msg.properties)
    return typing.GetPropertiesRes(status=status, properties=properties)


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.EvaluateIns:
    """Serialize `EvaluateIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns) -> typing.EvaluateIns:
    """Deserialize `EvaluateIns` from ProtoBuf."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes) -> ClientMessage.EvaluateRes:
    """Serialize `EvaluateRes` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.EVALUATE_NOT_IMPLEMENTED:
        return ClientMessage.EvaluateRes(status=status_msg)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.EvaluateRes(
        status=status_msg,
        loss=res.loss,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes) -> typing.EvaluateRes:
    """Deserialize `EvaluateRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        status=status,
        loss=msg.loss,
        num_examples=msg.num_examples,
        metrics=metrics,
    )


# === Status messages ===


def status_to_proto(status: typing.Status) -> Status:
    """Serialize `Status` to ProtoBuf."""
    code = Code.OK
    if status.code == typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        code = Code.GET_PROPERTIES_NOT_IMPLEMENTED
    if status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = Code.GET_PARAMETERS_NOT_IMPLEMENTED
    if status.code == typing.Code.FIT_NOT_IMPLEMENTED:
        code = Code.FIT_NOT_IMPLEMENTED
    if status.code == typing.Code.EVALUATE_NOT_IMPLEMENTED:
        code = Code.EVALUATE_NOT_IMPLEMENTED
    return Status(code=code, message=status.message)


def status_from_proto(msg: Status) -> typing.Status:
    """Deserialize `Status` from ProtoBuf."""
    code = typing.Code.OK
    if msg.code == Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        code = typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED
    if msg.code == Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED
    if msg.code == Code.FIT_NOT_IMPLEMENTED:
        code = typing.Code.FIT_NOT_IMPLEMENTED
    if msg.code == Code.EVALUATE_NOT_IMPLEMENTED:
        code = typing.Code.EVALUATE_NOT_IMPLEMENTED
    return typing.Status(code=code, message=msg.message)


# === Properties messages ===


def properties_to_proto(properties: typing.Properties) -> Any:
    """Serialize `Properties` to ProtoBuf."""
    proto = {}
    for key in properties:
        proto[key] = scalar_to_proto(properties[key])
    return proto


def properties_from_proto(proto: Any) -> typing.Properties:
    """Deserialize `Properties` from ProtoBuf."""
    properties = {}
    for k in proto:
        properties[k] = scalar_from_proto(proto[k])
    return properties


# === Metrics messages ===


def metrics_to_proto(metrics: typing.Metrics) -> Any:
    """Serialize `Metrics` to ProtoBuf."""
    proto = {}
    for key in metrics:
        proto[key] = scalar_to_proto(metrics[key])
    return proto


def metrics_from_proto(proto: Any) -> typing.Metrics:
    """Deserialize `Metrics` from ProtoBuf."""
    metrics = {}
    for k in proto:
        metrics[k] = scalar_from_proto(proto[k])
    return metrics


# === Scalar messages ===


def scalar_to_proto(scalar: typing.Scalar) -> Scalar:
    """Serialize `Scalar` to ProtoBuf."""
    if isinstance(scalar, bool):
        return Scalar(bool=scalar)

    if isinstance(scalar, bytes):
        return Scalar(bytes=scalar)

    if isinstance(scalar, float):
        return Scalar(double=scalar)

    if isinstance(scalar, int):
        return Scalar(sint64=scalar)

    if isinstance(scalar, str):
        return Scalar(string=scalar)

    raise Exception(
        f"Accepted types: {bool, bytes, float, int, str} (but not {type(scalar)})"
    )


def scalar_from_proto(scalar_msg: Scalar) -> typing.Scalar:
    """Deserialize `Scalar` from ProtoBuf."""
    scalar_field = scalar_msg.WhichOneof("scalar")
    scalar = getattr(scalar_msg, cast(str, scalar_field))
    return cast(typing.Scalar, scalar)


# === Value messages ===


_python_type_to_field_name = {
    float: "double",
    int: "sint64",
    bool: "bool",
    str: "string",
    bytes: "bytes",
}


_python_list_type_to_message_and_field_name = {
    float: (Value.DoubleList, "double_list"),
    int: (Value.Sint64List, "sint64_list"),
    bool: (Value.BoolList, "bool_list"),
    str: (Value.StringList, "string_list"),
    bytes: (Value.BytesList, "bytes_list"),
}


def _check_value(value: typing.Value) -> None:
    if isinstance(value, tuple(_python_type_to_field_name.keys())):
        return
    if isinstance(value, list):
        if len(value) > 0 and isinstance(
            value[0], tuple(_python_type_to_field_name.keys())
        ):
            data_type = type(value[0])
            for element in value:
                if isinstance(element, data_type):
                    continue
                raise Exception(
                    f"Inconsistent type: the types of elements in the list must "
                    f"be the same (expected {data_type}, but got {type(element)})."
                )
    else:
        raise TypeError(
            f"Accepted types: {bool, bytes, float, int, str} or "
            f"list of these types."
        )


def value_to_proto(value: typing.Value) -> Value:
    """Serialize `Value` to ProtoBuf."""
    _check_value(value)

    arg = {}
    if isinstance(value, list):
        msg_class, field_name = _python_list_type_to_message_and_field_name[
            type(value[0]) if len(value) > 0 else int
        ]
        arg[field_name] = msg_class(vals=value)
    else:
        arg[_python_type_to_field_name[type(value)]] = value
    return Value(**arg)


def value_from_proto(value_msg: Value) -> typing.Value:
    """Deserialize `Value` from ProtoBuf."""
    value_field = cast(str, value_msg.WhichOneof("value"))
    if value_field.endswith("list"):
        value = list(getattr(value_msg, value_field).vals)
    else:
        value = getattr(value_msg, value_field)
    return cast(typing.Value, value)


# === Named Values ===


def named_values_to_proto(
    named_values: Dict[str, typing.Value],
) -> Dict[str, Value]:
    """Serialize named values to ProtoBuf."""
    return {name: value_to_proto(value) for name, value in named_values.items()}


def named_values_from_proto(
    named_values_proto: MutableMapping[str, Value]
) -> Dict[str, typing.Value]:
    """Deserialize named values from ProtoBuf."""
    return {name: value_from_proto(value) for name, value in named_values_proto.items()}
