#ifndef TENSEAL_TENSOR_PLAINTEXTVECTOR_H
#define TENSEAL_TENSOR_PLAINTEXTVECTOR_H

#include "tenseal/cpp/tensors/encrypted_vector.h"
#include "tenseal/proto/tensors.pb.h"
#include <iostream>

namespace tenseal {

using namespace seal;
using namespace std;

/**
 * Holds a vector of real numbers in its encrypted form using the CKKS
 *homomorphic encryption scheme.
 **/
class PlaintextVector
    : public enable_shared_from_this<PlaintextVector> {
   public:
    using encrypted_t = shared_ptr<PlaintextVector>;

    template <typename... Args>
    static encrypted_t Create(Args&&... args) {
        return encrypted_t(new PlaintextVector(std::forward<Args>(args)...));
    }

    /**
     * Load/Save the vector from/to a serialized protobuffer.
     **/
    void load(const shared_ptr<SEALContext>& ctx, const string& vec);
    string save() const;

    /** using plain_t = PlainTensor<double>;
     *Recreates a new PlaintextVector from the current one, without any
     *pointer/reference to this one.
     **/
    encrypted_t copy() const;
    //encrypted_t deepcopy() const;

    const vector<Plaintext>& plaintext() const { return this->_plaintexts; }

  protected:
    std::vector<size_t> _sizes;
    std::vector<Plaintext> _plaintexts;
    //optional<string> _lazy_buffer;

   private:
    /*
    Private evaluation functions to process both scalar and vector arguments.
    */

    PlaintextVector(const vector<Plaintext>& vec);
    PlaintextVector(const shared_ptr<const PlaintextVector>& vec);
    PlaintextVector(const shared_ptr<TenSEALContext>& ctx, const string& vec);
    PlaintextVector(const TenSEALContextProto& ctx, const PlaintextVectorProto& vec);
    PlaintextVector(const shared_ptr<TenSEALContext>& ctx,
               const PlaintextVectorProto& vec);
    

    void load_proto(const shared_ptr<SEALContext>& ctx,const PlaintextVectorProto& buffer);
    PlaintextVectorProto save_proto() const;
};

}  // namespace tenseal

#endif
