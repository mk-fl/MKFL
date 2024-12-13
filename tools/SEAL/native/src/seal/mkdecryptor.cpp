// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/mkdecryptor.h"
#include "seal/valcheck.h"
#include "seal/util/common.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/polycore.h"
#include "seal/util/scalingvariant.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintcore.h"
#include <algorithm>
#include <stdexcept>

#include "seal/util/rlwe.h"

using namespace std;
using namespace seal::util;

namespace seal
{
    MKDecryptor::MKDecryptor(const SEALContext &context) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }
        auto &parms = context_.key_context_data()->parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
    }

    void MKDecryptor::decrypt(const Ciphertext &encrypted, Plaintext &destination)
    {
        // Verify that encrypted is valid.
        if (!is_valid_for(encrypted, context_))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        // Additionally check that ciphertext doesn't have trivial size
        if (encrypted.size() < SEAL_CIPHERTEXT_SIZE_MIN)
        {
            throw invalid_argument("encrypted is empty");
        }

        auto &context_data = *context_.first_context_data();
        auto &parms = context_data.parms();

        if(parms.scheme() != scheme_type::mk_ckks)
        {
            throw invalid_argument("unsupported scheme");
        }
        
        mk_ckks_decrypt(encrypted,destination,pool_);
        return;
        
    }

    void MKDecryptor::mk_ckks_decrypt(const Ciphertext &encrypted, Plaintext &destination, MemoryPoolHandle pool)
    {
        if (!encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted must be in NTT form");
        }

        // We already know that the parameters are valid
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t rns_poly_uint64_count = mul_safe(coeff_count, coeff_modulus_size);

        // Decryption consists in copying the c_0 term into the Plaintext destination
        // c_0 contains the C_sum_0 + DS_0 + ... + DS_i
        // DS_i is the decryption share of the i-th party

        // Since we overwrite destination, we zeroize destination parameters
        // This is necessary, otherwise resize will throw an exception.
        destination.parms_id() = parms_id_zero;

        // Resize destination to appropriate size
        destination.resize(rns_poly_uint64_count);

        // Copy c_0 into destination
        copy_ct_share(encrypted, RNSIter(destination.data(), coeff_count), pool);

        // Set destination parameters as in encrypted
        destination.parms_id() = encrypted.parms_id();
        destination.scale() = encrypted.scale();
    }

    // Copy c_0 + decryption share mod q into destination.
    // Store result in destination in RNS form.
    void MKDecryptor::copy_ct_share(const Ciphertext &encrypted, RNSIter destination, MemoryPoolHandle pool)
    {
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t key_coeff_modulus_size = context_.key_context_data()->parms().coeff_modulus().size();
        size_t encrypted_size = encrypted.size();
        auto is_ntt_form = encrypted.is_ntt_form();

        auto ntt_tables = context_data.small_ntt_tables();

        if (encrypted_size == 2)
        {
            ConstRNSIter c0(encrypted.data(0), coeff_count);
            ConstRNSIter c1(encrypted.data(1), coeff_count);
            if (is_ntt_form)
            {
                SEAL_ITERATE(
                    iter(c0, c1, coeff_modulus, destination), coeff_modulus_size, [&](auto I) {
                        // copy c_0 to the result; note that destination should be in the same (NTT) form as encrypted
                        copy_poly_coeffmod(get<0>(I), coeff_count, get<2>(I), get<3>(I));
                    });
            }
            else
            {
                SEAL_ITERATE(
                    iter(c0, c1, coeff_modulus, ntt_tables, destination), coeff_modulus_size,
                    [&](auto I) {
                         // copy c_0 to the result; note that destination should be in the same (NTT) form as encrypted
                        copy_poly_coeffmod(get<0>(I), coeff_count, get<2>(I), get<4>(I));
                    });
            }
        }
        else
        {
            // Finally copy c_0 to the result; note that destination should be in the same (NTT) form as encrypted
            copy_poly_coeffmod(*iter(encrypted), coeff_modulus_size, coeff_modulus, destination);
        }
    }

} // namespace seal
