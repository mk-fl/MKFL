"""Vector of values encrypted using CKKS. Less flexible, but more efficient than CKKSTensor.
"""
from typing import List
import tenseal as ts
from tenseal.tensors.abstract_tensor import AbstractTensor


class PlaintextVector(AbstractTensor):
    def __init__(
        self,
        vector=None,
        data: ts._ts_cpp.PlaintextVector = None,
    ):
        """Constructor method for the PlaintextVector object, which can store a vector of
        float numbers in encrypted form, using the CKKS homomorphic encryption scheme.

        Args:
            context: a Context object, holding the encryption parameters and keys.
            vector (of float): a vector holding data to be encrypted.
            scale: the scale to be used to encode vector values. PlaintextVector will use the global_scale provided by the context if it's set to None.
            data: A ts._ts_cpp.PlaintextVector to wrap. We won't construct a new object if it's passed.

        Returns:
            PlaintextVector object.
        """
        #TODO fix problem with context
        # wrapping
        if data is not None:
            self.data = data
        # constructing a new object
        else:
            
            #TODO check vector type
            self.data = ts._ts_cpp.PlaintextVector( vector)
    
    def scale(self) -> float:
        return None
        #return self.data.scale()
    def size(self) -> int:
        return None
        #return self.data.size()

    #@property
    #def shape(self) -> List[int]:
    #    return [self.size()]

    def plaintext(self) -> List["ts._ts_cpp.Plaintext"]:
        return self.data.plaintext()

    #@classmethod
    #def pack_vectors(cls, vectors: List["PlaintextVector"]) -> "PlaintextVector":
    #    to_pack = []
    #    for v in vectors:
    #        if not isinstance(v, cls):
    #            raise TypeError("vectors to pack must be of type tenseal.PlaintextVector")
    #        to_pack.append(v.data)
    #    return cls(data=ts._ts_cpp.PlaintextVector.pack_vectors(to_pack))

        









