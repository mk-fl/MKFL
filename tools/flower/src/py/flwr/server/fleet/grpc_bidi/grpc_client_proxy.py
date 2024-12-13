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
"""Flower ClientProxy implementation using gRPC bidirectional streaming."""


from typing import Optional, List, Tuple

from flwr import common
from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.client_proxy import ClientProxy
from flwr.server.fleet.grpc_bidi.grpc_bridge import GrpcBridge, InsWrapper, ResWrapper


class GrpcClientProxy(ClientProxy):
    """Flower ClientProxy that uses gRPC to delegate tasks over the network."""

    def __init__(
        self,
        cid: str,
        bridge: GrpcBridge,
    ):
        super().__init__(cid)
        self.bridge = bridge

    def get_properties(
        self,
        ins: common.GetPropertiesIns,
        timeout: Optional[float],
    ) -> common.GetPropertiesRes:
        """Request client's set of internal properties."""
        get_properties_msg = serde.get_properties_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_properties_ins=get_properties_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_properties_res = serde.get_properties_res_from_proto(
            client_msg.get_properties_res
        )
        return get_properties_res

    def get_parameters(
        self,
        ins: common.GetParametersIns,
        timeout: Optional[float],
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        get_parameters_msg = serde.get_parameters_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_parameters_ins=get_parameters_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_parameters_res = serde.get_parameters_res_from_proto(
            client_msg.get_parameters_res
        )
        return get_parameters_res

    def fit(
        self,
        ins: common.FitIns,
        timeout: Optional[float],
    ) -> common.FitRes:
        """Refine the provided parameters using the locally held dataset."""
        fit_ins_msg = serde.fit_ins_to_proto(ins)

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(fit_ins=fit_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        fit_res = serde.fit_res_from_proto(client_msg.fit_res)
        return fit_res
    
    def get_pk(
        self,
        context,
        timeout: Optional[float],
    ):
        """Refine the provided parameters using the locally held dataset."""
        get_pk_ins_msg = serde.get_pk_ins_to_proto(context)

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_pk_ins=get_pk_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_pk_res = serde.get_pk_res_from_proto(client_msg.get_pk_res)
        return get_pk_res
    
    def send_pk(
        self,
        ctx,
        timeout: Optional[float],
    ):
        """Refine the provided parameters using the locally held dataset."""
        send_pk_ins_msg = serde.send_pk_ins_to_proto(ctx)

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(send_pk_ins=send_pk_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        send_pk_res = serde.send_pk_res_from_proto(client_msg.send_pk_res)
        return send_pk_res
    
    def get_parms(
        self,
        timeout: Optional[float],
    ):
        """Refine the provided parameters using the locally held dataset."""
        get_parms_ins_msg = serde.get_parms_ins_to_proto()

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_parms_ins=get_parms_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_parms_res = serde.get_parms_res_from_proto(client_msg.get_parms_res)
        return get_parms_res
    
    def send_enc(
        self,
        ctx,
        enc,
        timeout: Optional[float],
    ):
        """Refine the provided parameters using the locally held dataset."""
        send_enc_ins_msg = serde.send_enc_ins_to_proto(ctx,enc=enc)

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(send_enc_ins=send_enc_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        send_enc_res = serde.send_enc_res_from_proto(client_msg.send_enc_res)
        return send_enc_res
    
    def send_ds(
        self,
        ctx,
        enc,
        timeout: Optional[float],
    ):
        """Refine the provided parameters using the locally held dataset."""
        send_ds_ins_msg = serde.send_ds_ins_to_proto(ctx, enc)

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(send_ds_ins=send_ds_ins_msg), 
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        send_ds_res = serde.send_ds_res_from_proto(client_msg.send_ds_res)
        return send_ds_res

    def evaluate_enc(
        self,
        ins: common.EvaluateIns,
        timeout: Optional[float],
    ) -> common.EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        evaluate_msg = serde.send_eval_ins_to_proto()
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(send_eval_ins=evaluate_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        evaluate_res = serde.send_eval_res_from_proto(client_msg.send_eval_res)
        return evaluate_res

    def evaluate(
        self,
        ins: common.EvaluateIns,
        timeout: Optional[float],
    ) -> common.EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        evaluate_msg = serde.evaluate_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(evaluate_ins=evaluate_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        evaluate_res = serde.evaluate_res_from_proto(client_msg.evaluate_res)
        return evaluate_res

    def reconnect(
        self,
        ins: common.ReconnectIns,
        timeout: Optional[float],
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        reconnect_ins_msg = serde.reconnect_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(reconnect_ins=reconnect_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        disconnect = serde.disconnect_res_from_proto(client_msg.disconnect_res)
        return disconnect
    
    def request(self, question: str, l: List[int]) -> Tuple[str, int]:
        request_msg = serde.example_msg_to_proto(question, l)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(send_sum_ins=request_msg),
                timeout=10,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        response, answer = serde.example_res_from_proto(client_msg.send_val_res)
        return response, answer
        
    def get_gradients(
        self,
        timeout: Optional[float],
    ):
        get_gradients_ins_msg = serde.get_gradients_ins_to_proto()

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_gradients_ins=get_gradients_ins_msg),
                timeout=timeout,
            )
        )

        client_msg: ClientMessage = res_wrapper.client_message
        get_gradients_res = serde.get_gradients_res_from_proto(client_msg.get_gradients_res)

        return get_gradients_res
        
    def identify(
        self,
        timeout: Optional[float],
    ):
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(identify_ins=ServerMessage.IdentifyIns()),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        status = client_msg.identify_res.status
        return status
        
    def get_contributions(
        self,
        ins,
        timeout: Optional[float],
    ):
        get_contributions_ins_msg = serde.get_contributions_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_contributions_ins=get_contributions_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_contributions_res = serde.get_contributions_res_from_proto(client_msg.get_contributions_res)
        return get_contributions_res
    
    def send_public_key(
        self,
        ins,
        timeout: Optional[float],
    ):
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(send_public_key_ins=ServerMessage.SendPublicKeyIns(publickey=ins)),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        return 0
        
