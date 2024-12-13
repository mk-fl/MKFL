#include "tenseal/cpp/tensors/plaintextvector.h"

using namespace seal;
using namespace std;

namespace tenseal {

PlaintextVector::PlaintextVector(const vector<Plaintext>& vec) {
    

    if (vec.empty()) {
        throw invalid_argument("Attempting to encrypt an empty vector");
    }
    this->_plaintexts = vec;
    
}

PlaintextVector::PlaintextVector(const shared_ptr<const PlaintextVector>& vec) {
    this->_plaintexts = vec->plaintext();
}

PlaintextVector::PlaintextVector(const shared_ptr<TenSEALContext>& ctx,
                       const string& vec) {
    this->load(ctx->seal_context(),vec);
}

PlaintextVector::PlaintextVector(const TenSEALContextProto& ctx,
                       const PlaintextVectorProto& vec) {
    shared_ptr<TenSEALContext> c = TenSEALContext::Create(ctx);
    this->load_proto(c->seal_context(), vec);
}

PlaintextVector::PlaintextVector(const shared_ptr<TenSEALContext>& ctx,
                       const PlaintextVectorProto& vec) {
    this->load_proto(ctx->seal_context(), vec);
    cout << "constructor 5" <<endl;
}







void PlaintextVector::load_proto(const shared_ptr<SEALContext>& ctx, const PlaintextVectorProto& vec) {
    if (ctx == nullptr) {
        throw invalid_argument("context missing for deserialization");
    }

    this->_sizes = vector<size_t>();
    this->_plaintexts = vector<Plaintext>();

    for (auto& sz : vec.sizes()) this->_sizes.push_back(sz);
    for (auto& ct : vec.plaintexts())
        this->_plaintexts.push_back(SEALDeserialize<Plaintext>(
            *ctx.get(), ct));
}

PlaintextVectorProto PlaintextVector::save_proto() const {
    PlaintextVectorProto buffer;

    for (auto& ct : this->_plaintexts) {
        buffer.add_plaintexts(SEALSerialize<Plaintext>(ct));
    }
    for (auto& sz : this->_sizes) {
        buffer.add_sizes(sz);
    }
    return buffer;
}

void PlaintextVector::load(const shared_ptr<SEALContext>& ctx ,const std::string& vec) {

    PlaintextVectorProto buffer;
    if (!buffer.ParseFromArray(vec.c_str(), static_cast<int>(vec.size()))) {
        throw invalid_argument("failed to parse CKKS stream");
    }
    this->load_proto(ctx,buffer);
}

std::string PlaintextVector::save() const {
    auto buffer = this->save_proto();
    std::string output;
    output.resize(proto_bytes_size(buffer));

    if (!buffer.SerializeToArray((void*)output.c_str(),
                                 static_cast<int>(proto_bytes_size(buffer)))) {
        throw invalid_argument("failed to save CKKS proto");
    }

    return output;
}

shared_ptr<PlaintextVector> PlaintextVector::copy() const {

    return shared_ptr<PlaintextVector>(new PlaintextVector(shared_from_this()));
}

//shared_ptr<PlaintextVector> PlaintextVector::deepcopy() const {
//
//    this->plaintexts()[0].data(0)
//
//    TenSEALContextProto ctx = this->tenseal_context()->save_proto(
//        /*save_public_key=*/true, /*save_secret_key=*/true,
//        /*save_galois_keys=*/true, /*save_relin_keys=*/true);
//    PlaintextVectorProto vec = this->save_proto();
//    return PlaintextVector::Create(vec);
//}

}  // namespace tenseal
