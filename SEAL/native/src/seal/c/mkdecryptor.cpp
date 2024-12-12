// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// SEALNet
#include "seal/c/mkdecryptor.h"
#include "seal/c/utilities.h"

// SEAL
#include "seal/mkdecryptor.h"

using namespace std;
using namespace seal;
using namespace seal::c;

SEAL_C_FUNC MKDecryptor_Create(void *context, void *secret_key, void **decryptor)
{
    const SEALContext *ctx = FromVoid<SEALContext>(context);
    IfNullRet(ctx, E_POINTER);
    SecretKey *secretKey = FromVoid<SecretKey>(secret_key);
    IfNullRet(secretKey, E_POINTER);
    IfNullRet(decryptor, E_POINTER);

    try
    {
        MKDecryptor *decr = new MKDecryptor(*ctx, *secretKey);
        *decryptor = decr;
        return S_OK;
    }
    catch (const invalid_argument &)
    {
        return E_INVALIDARG;
    }
}

SEAL_C_FUNC MKDecryptor_Destroy(void *thisptr)
{
    MKDecryptor *decryptor = FromVoid<MKDecryptor>(thisptr);
    IfNullRet(decryptor, E_POINTER);

    delete decryptor;
    return S_OK;
}

SEAL_C_FUNC MKDecryptor_Decrypt(void *thisptr, void *encrypted, void *destination)
{
    MKDecryptor *decryptor = FromVoid<MKDecryptor>(thisptr);
    IfNullRet(decryptor, E_POINTER);
    Ciphertext *encryptedptr = FromVoid<Ciphertext>(encrypted);
    IfNullRet(encryptedptr, E_POINTER);
    Plaintext *destinationptr = FromVoid<Plaintext>(destination);
    IfNullRet(destinationptr, E_POINTER);

    try
    {
        decryptor->decrypt(*encryptedptr, *destinationptr);
        return S_OK;
    }
    catch (const invalid_argument &)
    {
        return E_INVALIDARG;
    }
}
