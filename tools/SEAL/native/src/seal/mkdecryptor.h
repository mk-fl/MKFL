// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/ciphertext.h"
#include "seal/context.h"
#include "seal/encryptionparams.h"
#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/plaintext.h"
#include "seal/randomgen.h"
#include "seal/secretkey.h"
#include "seal/util/defines.h"
#include "seal/util/iterator.h"
#include "seal/util/locks.h"
#include "seal/util/ntt.h"
#include "seal/util/rns.h"

namespace seal
{
    /**
    Decrypts Ciphertext objects into Plaintext objects. Constructing a Decryptor
    requires a SEALContext with valid encryption parameters, and the secret key.
    The Decryptor is also used to compute the invariant noise budget in a given
    ciphertext.

    @par Overloads
    For the decrypt function we provide two overloads concerning the memory pool
    used in allocations needed during the operation. In one overload the global
    memory pool is used for this purpose, and in another overload the user can
    supply a MemoryPoolHandle to be used instead. This is to allow one single
    Decryptor to be used concurrently by several threads without running into
    thread contention in allocations taking place during operations. For example,
    one can share one single Decryptor across any number of threads, but in each
    thread call the decrypt function by giving it a thread-local MemoryPoolHandle
    to use. It is important for a developer to understand how this works to avoid
    unnecessary performance bottlenecks.


    @par NTT form
    When using the BFV scheme (scheme_type::bfv), all plaintext and ciphertexts
    should remain by default in the usual coefficient representation, i.e. not in
    NTT form. When using the CKKS/MK_CKKS scheme (scheme_type::ckks/scheme_type::mk_ckks), all plaintexts and
    ciphertexts should remain by default in NTT form. We call these scheme-specific
    NTT states the "default NTT form". Decryption requires the input ciphertexts
    to be in the default NTT form, and will throw an exception if this is not the
    case.
    */
    class MKDecryptor
    {
    public:
        /**
        Creates a Decryptor instance initialized with the specified SEALContext
        and secret key.

        @param[in] context The SEALContext
        @throws std::invalid_argument if the encryption parameters are not valid
        */
        MKDecryptor(const SEALContext &context);

        
        /*
        Decrypts a Ciphertext and stores the result in the destination parameter.

        @param[in] encrypted The ciphertext to decrypt
        @param[out] destination The plaintext to overwrite with the decrypted
        ciphertext
        @throws std::invalid_argument if encrypted is not valid for the encryption
        parameters
        @throws std::invalid_argument if encrypted is not in the default NTT form
        */
        void decrypt(const Ciphertext &encrypted, Plaintext &destination);

    private:
        void mk_ckks_decrypt(const Ciphertext &encrypted, Plaintext &destination, MemoryPoolHandle pool);

        MKDecryptor(const MKDecryptor &copy) = delete;

        MKDecryptor(MKDecryptor &&source) = delete;

        MKDecryptor &operator=(const MKDecryptor &assign) = delete;

        MKDecryptor &operator=(MKDecryptor &&assign) = delete;

        // Copy C_sum_0 + decryption shares mod q.
        // Store result in destination in RNS form.
        // destination has the size of an RNS polynomial.
        void copy_ct_share(const Ciphertext &encrypted, util::RNSIter destination, MemoryPoolHandle pool);

        // We use a fresh memory pool with `clear_on_destruction' enabled.
        MemoryPoolHandle pool_ = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

        SEALContext context_;

    };
} // namespace seal
