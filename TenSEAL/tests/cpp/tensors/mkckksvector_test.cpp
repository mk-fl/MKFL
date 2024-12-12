#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tenseal/cpp/tenseal.h"

namespace tenseal {
namespace {

using namespace ::testing;
using namespace std;

template <class Iterable>
bool are_close(const Iterable& l, const std::vector<double>& r) {
    if (l.size() != r.size()) {
        return false;
    }
    for (size_t idx = 0; idx < l.size(); ++idx) {
        if (std::abs(l[idx] - r[idx]) > 0.5) {
            return false;}
    }
    return true;
}

auto duplicate(shared_ptr<CKKSVector> in) {
    auto vec = in->save();

    return CKKSVector::Create(in->tenseal_context(), vec);
}

class MKCKKSVectorTest
    : public TestWithParam<tuple</*serialize_first=*/bool,
                                 /*encryption_type=*/encryption_type>> {
   protected:
    void SetUp() {}
};
TEST_P(MKCKKSVectorTest, TestCreateMKCKKS) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{


        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        auto l = CKKSVector::Create(ctx, std::vector<double>{1, 2, 3}, 1);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_THAT(l->ciphertext_size(), ElementsAreArray({2}));
    }
}

TEST_P(MKCKKSVectorTest, TestCreateMKCKKSFail) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{


        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        EXPECT_THROW(
            auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3})),
            std::exception);
    }
}

TEST_P(MKCKKSVectorTest, TestMKCKKSAdd) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{

        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        ctx->global_scale(std::pow(2, 40));

        ctx->auto_relin(false);
        ctx->auto_rescale(false);
        ctx->auto_mod_switch(false);
        auto sk = ctx->secret_key();

        auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
        auto r = CKKSVector::Create(ctx, std::vector<double>({3, 4, 4}));

    

        auto add = l->add(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_THAT(add->ciphertext_size(), ElementsAreArray({2}));

        auto ds= add->decryption_share(ctx,sk);
        vector<vector<Plaintext>> decr{ds};
        auto dec = add->mk_decrypt(decr);
        ASSERT_TRUE(are_close(dec.data(), {4, 6, 7}));

        l->add_inplace(r);
        l->add_inplace(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_THAT(l->ciphertext_size(), ElementsAreArray({2}));
        auto ds2= l->decryption_share(ctx,sk);
        //auto decr2 = l->add_share(ds2)->mk_decrypt();
        vector<vector<Plaintext>> decr2{ds2};
        auto dec2 = l->mk_decrypt(decr2);
        //decr = l->decrypt();
        ASSERT_TRUE(are_close(dec2.data(), {7, 10, 11}));
    }
}


TEST_P(MKCKKSVectorTest, TestMKCKKSPK) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());

    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{

        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        ctx->global_scale(std::pow(2, 40));

        ctx->auto_relin(false);
        ctx->auto_rescale(false);
        ctx->auto_mod_switch(false);

        auto pk = ctx.get()->public_key();
        auto sk = ctx->secret_key();
        

        auto ctx2 = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                        {60, 40, 40, 60}, enc_type,NULL,*pk);
        ASSERT_TRUE(ctx2 != nullptr);

        ctx2->global_scale(std::pow(2, 40));

        ctx2->auto_relin(false);
        ctx2->auto_rescale(false);
        ctx2->auto_mod_switch(false);

        auto pk2 = ctx2->public_key();
        auto sk2 = ctx2->secret_key();
        auto pk_sum = CKKSVector::Create(ctx,pk)->add(CKKSVector::Create(ctx2,pk2),true);
        auto pk_agg = PublicKey(pk_sum->ciphertext()[0]);
        ctx->set_publickey(pk_agg);
        ctx2->set_publickey(pk_agg);

        auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
        auto r = CKKSVector::Create(ctx2, std::vector<double>({3, 4, 4}));

        auto add = l->add(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_THAT(add->ciphertext_size(), ElementsAreArray({2}));

        auto ds= add->decryption_share(ctx,sk);
        auto ds2=add->decryption_share(ctx2,sk2);
        //auto decr = add->add_share(ds)->add_share(ds2)->mk_decrypt();
        vector<vector<Plaintext>> decr{ds,ds2};
        auto dec = add->mk_decrypt(decr);
        ASSERT_TRUE(are_close(dec.data(), {4, 6, 7}));

        l->add_inplace(r);
        l->add_inplace(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_THAT(l->ciphertext_size(), ElementsAreArray({2}));
        auto ds3= l->decryption_share(ctx, sk);
        auto ds4 = l->decryption_share(ctx2,sk2);
        //auto decr2 = l->add_share(ds3)->add_share(ds4)->mk_decrypt();
        vector<vector<Plaintext>> decr2{ds3,ds4};
        auto dec2 = l->mk_decrypt(decr2);
        //decr = l->decrypt();
        ASSERT_TRUE(are_close(dec2.data(), {7, 10, 11}));
    }
}

TEST_P(MKCKKSVectorTest, TestMKCKKSReplicateFirstSlot) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{


        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        ctx->generate_galois_keys();
        ctx->global_scale(std::pow(2, 40));
        auto sk = ctx->secret_key();

        auto vec = CKKSVector::Create(ctx, std::vector<double>({1}));
        auto replicated_vec = vec->replicate_first_slot(4);

        if (should_serialize_first) {
            replicated_vec = duplicate(replicated_vec);
        }
        auto ds = replicated_vec->decryption_share(ctx,sk);
        //auto result = replicated_vec->add_share(ds)->mk_decrypt();
        vector<vector<Plaintext>> decr{ds};
        auto result = replicated_vec->mk_decrypt(decr);
        ASSERT_EQ(result.size(), 4);
        ASSERT_TRUE(are_close(result.data(), {1, 1, 1, 1}));

        vec->mul_plain_inplace(2);
        vec->replicate_first_slot_inplace(6);
        auto ds2 = vec->decryption_share(ctx,sk);
        //result = vec->add_share(ds2)->mk_decrypt();
        vector<vector<Plaintext>> decr2{ds2};
        result = vec->mk_decrypt(decr2);
        ASSERT_EQ(result.size(), 6);
        ASSERT_TRUE(are_close(result.data(), {2, 2, 2, 2, 2, 2}));
    }
}

TEST_P(MKCKKSVectorTest, TestEmptyPlaintextMK) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{

        auto ctx =
            TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60,40,40,60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        EXPECT_THROW(CKKSVector::Create(ctx, std::vector<double>({})),
                    std::exception);
    }
}

TEST_F(MKCKKSVectorTest, TestMKCKKSVectorSerializationSize) {
    vector<double> input;
    for (double val = 0.5; val < 1000; ++val) input.push_back(val);
    

    auto pk_ctx =
        TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60, 40, 40, 60},
                            encryption_type::asymmetric);
    pk_ctx->global_scale(std::pow(2, 40));
    auto pk_vector = CKKSVector::Create(pk_ctx, input);

    /*auto sym_ctx =
        TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60, 40, 40, 60},
                            encryption_type::symmetric);
    sym_ctx->global_scale(std::pow(2, 40));
    auto sym_vector = CKKSVector::Create(sym_ctx, input);*/

    auto pk_buffer = pk_vector->save();
    //auto sym_buffer = sym_vector->save();

    fprintf(stderr, "pk_buffer size = %ld \n",
            pk_buffer.size());//, sym_buffer.size());
    //ASSERT_TRUE(pk_buffer.size() != sym_buffer.size());
    //ASSERT_TRUE(2 * sym_buffer.size() > pk_buffer.size());

}

TEST_P(MKCKKSVectorTest, TestMKCKKSAddBigVector) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());

    int poly_mod = 8192;
    int input_size = 100000;
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{

        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, poly_mod, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        ctx->global_scale(std::pow(2, 40));

        ctx->auto_relin(false);
        ctx->auto_rescale(false);
        ctx->auto_mod_switch(false);
        auto sk = ctx->secret_key();

        vector<double> l_input, r_input, expected;
        for (double i = 1.3; i < input_size; i++) {
            l_input.push_back(2 * i);
            r_input.push_back(3 * i);
            expected.push_back(5 * i);
        }

        auto l = CKKSVector::Create(ctx, l_input);
        auto r = CKKSVector::Create(ctx, r_input);

        auto add = l->add(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_EQ(add->chunked_size().size(), 2 * int(input_size / poly_mod) + 1);
        auto ds= add->decryption_share(ctx,sk);
        //auto decr = add->add_share(ds)->mk_decrypt();
        vector<vector<Plaintext>> decr{ds};
        auto dec = add->mk_decrypt(decr);
        ASSERT_TRUE(are_close(dec.data(), expected));
    }
}

TEST_P(MKCKKSVectorTest, TestMKCKKSAddBigVectorPK) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());

    int poly_mod = 8192;
    int input_size = 100000;
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{

        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, poly_mod, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        ctx->global_scale(std::pow(2, 40));

        ctx->auto_relin(false);
        ctx->auto_rescale(false);
        ctx->auto_mod_switch(false);
        auto sk = ctx->secret_key();
        auto pk = ctx->public_key();

        auto ctx2 = TenSEALContext::Create(scheme_type::mk_ckks, poly_mod, -1,
                                        {60, 40, 40, 60}, enc_type,NULL,*pk);
        ASSERT_TRUE(ctx2 != nullptr);
        ctx2->global_scale(std::pow(2, 40));
        ctx2->auto_relin(false);
        ctx2->auto_rescale(false);
        ctx2->auto_mod_switch(false);
        auto sk2 = ctx2->secret_key();
        auto pk2 = ctx2->public_key();

        vector<double> l_input, r_input, expected;
        for (double i = 1.3; i < input_size; i++) {
            l_input.push_back(2 * i);
            r_input.push_back(3 * i);
            expected.push_back(5 * i);
        }
        auto pk_sum_vector = CKKSVector::Create(ctx,pk)->add(CKKSVector::Create(ctx2,pk2),true);
        auto pk_sum = PublicKey(pk_sum_vector->ciphertext()[0]);

        ctx->set_publickey(pk_sum);
        ctx2->set_publickey(pk_sum);

        auto l = CKKSVector::Create(ctx, l_input);
        auto r = CKKSVector::Create(ctx2, r_input);

        auto add = l->add(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_EQ(add->chunked_size().size(), 2 * int(input_size / poly_mod) + 1);
        auto ds= add->decryption_share(ctx,sk);
        auto ds2= add->decryption_share(ctx2,sk2);
        //auto decr = add->add_share(ds)->add_share(ds2)->mk_decrypt();
        vector<vector<Plaintext>> decr{ds,ds2};
        auto dec = add->mk_decrypt(decr);
        ASSERT_TRUE(are_close(dec.data(), expected));
    }
}


INSTANTIATE_TEST_CASE_P(
    TestMKCKKSVector, MKCKKSVectorTest,
    ::testing::Values(make_tuple(false, encryption_type::asymmetric),
                      make_tuple(true, encryption_type::asymmetric),
                      make_tuple(false, encryption_type::symmetric),
                      make_tuple(true, encryption_type::symmetric)));

TEST_F(MKCKKSVectorTest, TestMKCKKSLazyContext) {
    auto ctx =
        TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60, 40, 40, 60});
    ASSERT_TRUE(ctx != nullptr);

    ctx->global_scale(std::pow(2, 40));
    auto sk = ctx->secret_key();

    auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
    auto r = CKKSVector::Create(ctx, std::vector<double>({3, 4, 4}));

    auto buffer = l->save();
    auto newl = CKKSVector::Create(buffer);

    EXPECT_THROW(newl->add(r), std::exception);

    newl->link_tenseal_context(ctx);
    auto res = newl->add(r);
    auto ds = res->decryption_share(ctx,sk);
    //auto decr = res->add_share(ds)->mk_decrypt();
    vector<vector<Plaintext>> decr{ds};
    auto dec = res->mk_decrypt(decr);
    ASSERT_TRUE(are_close(dec.data(), {4, 6, 7}));
}

TEST_F(MKCKKSVectorTest, TestMKCKKSLazyContextSanityDoubleSerde) {
    auto ctx =
        TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60, 40, 40, 60});
    ASSERT_TRUE(ctx != nullptr);

    ctx->global_scale(std::pow(2, 40));
    auto sk = ctx->secret_key();

    auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
    auto r = CKKSVector::Create(ctx, std::vector<double>({3, 4, 4}));

    // double serde
    auto buffer = l->save();
    auto newl = CKKSVector::Create(buffer);
    buffer = newl->save();
    newl = CKKSVector::Create(buffer);

    newl->link_tenseal_context(ctx);
    auto res = newl->add(r);
    auto ds = res->decryption_share(ctx,sk);
    //auto decr = res->add_share(ds)->mk_decrypt();
    vector<vector<Plaintext>> decr{ds};
    auto dec = res->mk_decrypt(decr);
    ASSERT_TRUE(are_close(dec.data(), {4, 6, 7}));
}

TEST_F(MKCKKSVectorTest, TestMKCKKSLazyContextSanityCopy) {
    auto ctx =
        TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60, 40, 40, 60});
    ASSERT_TRUE(ctx != nullptr);

    ctx->global_scale(std::pow(2, 40));
    auto sk = ctx->secret_key();

    auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
    auto r = CKKSVector::Create(ctx, std::vector<double>({3, 4, 4}));

    auto buffer = l->save();
    auto newl = CKKSVector::Create(buffer);

    auto cpy = newl->copy();
    cpy->link_tenseal_context(ctx);
    auto res = cpy->add(r);
    auto ds = res->decryption_share(ctx,sk);
    //auto decr = res->add_share(ds)->decrypt();
    vector<vector<Plaintext>> decr{ds};
    auto dec = res->mk_decrypt(decr);
    ASSERT_TRUE(are_close(dec.data(), {4, 6, 7}));
}

TEST_F(MKCKKSVectorTest, TestMKCKKSLazyContextSanityDeepcopy) {
    auto ctx =
        TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60, 40, 40, 60});
    ASSERT_TRUE(ctx != nullptr);

    ctx->global_scale(std::pow(2, 40));
    auto sk = ctx->secret_key();

    auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
    auto r = CKKSVector::Create(ctx, std::vector<double>({3, 4, 4}));

    auto buffer = l->save();
    auto newl = CKKSVector::Create(buffer);

    auto cpy = newl->deepcopy();
    cpy->link_tenseal_context(ctx);
    auto res = cpy->add(r);
    auto ds = res->decryption_share(ctx,sk);
    //auto decr = res->add_share(ds)->decrypt();
    vector<vector<Plaintext>> decr{ds};
    auto dec = res->mk_decrypt(decr);
    ASSERT_TRUE(are_close(dec.data(), {4, 6, 7}));
}

//DECODE


TEST_P(MKCKKSVectorTest, TestMKCKKSAddDecode) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{

        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        ctx->global_scale(std::pow(2, 40));

        ctx->auto_relin(false);
        ctx->auto_rescale(false);
        ctx->auto_mod_switch(false);
        auto sk = ctx->secret_key();

        auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
        auto r = CKKSVector::Create(ctx, std::vector<double>({3, 4, 4}));

    

        auto add = l->add(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_THAT(add->ciphertext_size(), ElementsAreArray({2}));

        auto ds= add->decryption_share(ctx,sk);
        auto decr = add->add_share(ds)->mk_decode();
        ASSERT_TRUE(are_close(decr.data(), {4, 6, 7}));

        l->add_inplace(r);
        l->add_inplace(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_THAT(l->ciphertext_size(), ElementsAreArray({2}));
        auto ds2= l->decryption_share(ctx,sk);
        auto decr2 = l->add_share(ds2)->mk_decode();
        ASSERT_TRUE(are_close(decr2.data(), {7, 10, 11}));
    }
}


TEST_P(MKCKKSVectorTest, TestMKCKKSPKDecode) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());

    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{

        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        ctx->global_scale(std::pow(2, 40));

        ctx->auto_relin(false);
        ctx->auto_rescale(false);
        ctx->auto_mod_switch(false);

        auto pk = ctx.get()->public_key();
        auto sk = ctx->secret_key();
        

        auto ctx2 = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                        {60, 40, 40, 60}, enc_type,NULL,*pk);
        ASSERT_TRUE(ctx2 != nullptr);

        ctx2->global_scale(std::pow(2, 40));

        ctx2->auto_relin(false);
        ctx2->auto_rescale(false);
        ctx2->auto_mod_switch(false);

        auto pk2 = ctx2->public_key();
        auto sk2 = ctx2->secret_key();
        auto pk_sum = CKKSVector::Create(ctx,pk)->add(CKKSVector::Create(ctx2,pk2),true);
        auto pk_agg = PublicKey(pk_sum->ciphertext()[0]);
        ctx->set_publickey(pk_agg);
        ctx2->set_publickey(pk_agg);

        auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
        auto r = CKKSVector::Create(ctx2, std::vector<double>({3, 4, 4}));

        auto add = l->add(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_THAT(add->ciphertext_size(), ElementsAreArray({2}));

        auto ds= add->decryption_share(ctx,sk);
        auto ds2=add->decryption_share(ctx2,sk2);
        auto decr = add->add_share(ds)->add_share(ds2)->mk_decode();
        ASSERT_TRUE(are_close(decr.data(), {4, 6, 7}));

        l->add_inplace(r);
        l->add_inplace(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_THAT(l->ciphertext_size(), ElementsAreArray({2}));
        auto ds3= l->decryption_share(ctx, sk);
        auto ds4 = l->decryption_share(ctx2,sk2);
        auto decr2 = l->add_share(ds3)->add_share(ds4)->mk_decode();
        ASSERT_TRUE(are_close(decr2.data(), {7, 10, 11}));
    }
}

TEST_P(MKCKKSVectorTest, TestMKCKKSReplicateFirstSlotDecode) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{


        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        ctx->generate_galois_keys();
        ctx->global_scale(std::pow(2, 40));
        auto sk = ctx->secret_key();

        auto vec = CKKSVector::Create(ctx, std::vector<double>({1}));
        auto replicated_vec = vec->replicate_first_slot(4);

        if (should_serialize_first) {
            replicated_vec = duplicate(replicated_vec);
        }
        auto ds = replicated_vec->decryption_share(ctx,sk);
        auto result = replicated_vec->add_share(ds)->mk_decode();
        ASSERT_EQ(result.size(), 4);
        ASSERT_TRUE(are_close(result.data(), {1, 1, 1, 1}));

        vec->mul_plain_inplace(2);
        vec->replicate_first_slot_inplace(6);
        auto ds2 = vec->decryption_share(ctx,sk);
        result = vec->add_share(ds2)->mk_decode();
        ASSERT_EQ(result.size(), 6);
        ASSERT_TRUE(are_close(result.data(), {2, 2, 2, 2, 2, 2}));
    }
} 

TEST_P(MKCKKSVectorTest, TestMKCKKSAddBigVectorDecode) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());

    int poly_mod = 8192;
    int input_size = 100000;
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{

        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, poly_mod, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        ctx->global_scale(std::pow(2, 40));

        ctx->auto_relin(false);
        ctx->auto_rescale(false);
        ctx->auto_mod_switch(false);
        auto sk = ctx->secret_key();

        vector<double> l_input, r_input, expected;
        for (double i = 1.3; i < input_size; i++) {
            l_input.push_back(2 * i);
            r_input.push_back(3 * i);
            expected.push_back(5 * i);
        }

        auto l = CKKSVector::Create(ctx, l_input);
        auto r = CKKSVector::Create(ctx, r_input);

        auto add = l->add(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_EQ(add->chunked_size().size(), 2 * int(input_size / poly_mod) + 1);
        auto ds= add->decryption_share(ctx,sk);
        auto decr = add->add_share(ds)->mk_decode();
        ASSERT_TRUE(are_close(decr.data(), expected));
    }
}

TEST_P(MKCKKSVectorTest, TestMKCKKSAddBigVectorPKDecode) {
    auto should_serialize_first = get<0>(GetParam());
    auto enc_type = get<1>(GetParam());

    int poly_mod = 8192;
    int input_size = 100000;
    if (enc_type==encryption_type::symmetric){
        EXPECT_THROW(auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1,
                                      {60, 40, 40, 60}, enc_type), std::exception);
    }
    else{

        auto ctx = TenSEALContext::Create(scheme_type::mk_ckks, poly_mod, -1,
                                        {60, 40, 40, 60}, enc_type);
        ASSERT_TRUE(ctx != nullptr);

        ctx->global_scale(std::pow(2, 40));

        ctx->auto_relin(false);
        ctx->auto_rescale(false);
        ctx->auto_mod_switch(false);
        auto sk = ctx->secret_key();
        auto pk = ctx->public_key();

        auto ctx2 = TenSEALContext::Create(scheme_type::mk_ckks, poly_mod, -1,
                                        {60, 40, 40, 60}, enc_type,NULL,*pk);
        ASSERT_TRUE(ctx2 != nullptr);
        ctx2->global_scale(std::pow(2, 40));
        ctx2->auto_relin(false);
        ctx2->auto_rescale(false);
        ctx2->auto_mod_switch(false);
        auto sk2 = ctx2->secret_key();
        auto pk2 = ctx2->public_key();

        vector<double> l_input, r_input, expected;
        for (double i = 1.3; i < input_size; i++) {
            l_input.push_back(2 * i);
            r_input.push_back(3 * i);
            expected.push_back(5 * i);
        }
        auto pk_sum_vector = CKKSVector::Create(ctx,pk)->add(CKKSVector::Create(ctx2,pk2),true);
        auto pk_sum = PublicKey(pk_sum_vector->ciphertext()[0]);

        ctx->set_publickey(pk_sum);
        ctx2->set_publickey(pk_sum);

        auto l = CKKSVector::Create(ctx, l_input);
        auto r = CKKSVector::Create(ctx2, r_input);

        auto add = l->add(r);

        if (should_serialize_first) {
            l = duplicate(l);
        }

        ASSERT_EQ(add->chunked_size().size(), 2 * int(input_size / poly_mod) + 1);
        auto ds= add->decryption_share(ctx,sk);
        auto ds2= add->decryption_share(ctx2,sk2);
        auto decr = add->add_share(ds)->add_share(ds2)->mk_decode();
        ASSERT_TRUE(are_close(decr.data(), expected));
    }
}


TEST_F(MKCKKSVectorTest, TestMKCKKSLazyContextDecode) {
    auto ctx =
        TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60, 40, 40, 60});
    ASSERT_TRUE(ctx != nullptr);

    ctx->global_scale(std::pow(2, 40));
    auto sk = ctx->secret_key();

    auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
    auto r = CKKSVector::Create(ctx, std::vector<double>({3, 4, 4}));

    auto buffer = l->save();
    auto newl = CKKSVector::Create(buffer);

    EXPECT_THROW(newl->add(r), std::exception);

    newl->link_tenseal_context(ctx);
    auto res = newl->add(r);
    auto ds = res->decryption_share(ctx,sk);
    auto decr = res->add_share(ds)->mk_decode();
    ASSERT_TRUE(are_close(decr.data(), {4, 6, 7}));
}

TEST_F(MKCKKSVectorTest, TestMKCKKSLazyContextSanityDoubleSerdeDecode) {
    auto ctx =
        TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60, 40, 40, 60});
    ASSERT_TRUE(ctx != nullptr);

    ctx->global_scale(std::pow(2, 40));
    auto sk = ctx->secret_key();

    auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
    auto r = CKKSVector::Create(ctx, std::vector<double>({3, 4, 4}));

    // double serde
    auto buffer = l->save();
    auto newl = CKKSVector::Create(buffer);
    buffer = newl->save();
    newl = CKKSVector::Create(buffer);

    newl->link_tenseal_context(ctx);
    auto res = newl->add(r);
    auto ds = res->decryption_share(ctx,sk);
    auto decr = res->add_share(ds)->mk_decode();
    ASSERT_TRUE(are_close(decr.data(), {4, 6, 7}));
}

TEST_F(MKCKKSVectorTest, TestMKCKKSLazyContextSanityCopyDecode) {
    auto ctx =
        TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60, 40, 40, 60});
    ASSERT_TRUE(ctx != nullptr);

    ctx->global_scale(std::pow(2, 40));
    auto sk = ctx->secret_key();

    auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
    auto r = CKKSVector::Create(ctx, std::vector<double>({3, 4, 4}));

    auto buffer = l->save();
    auto newl = CKKSVector::Create(buffer);

    auto cpy = newl->copy();
    cpy->link_tenseal_context(ctx);
    auto res = cpy->add(r);
    auto ds = res->decryption_share(ctx,sk);
    auto decr = res->add_share(ds)->mk_decode();
    ASSERT_TRUE(are_close(decr.data(), {4, 6, 7}));
}

TEST_F(MKCKKSVectorTest, TestMKCKKSLazyContextSanityDeepcopyDecode) {
    auto ctx =
        TenSEALContext::Create(scheme_type::mk_ckks, 8192, -1, {60, 40, 40, 60});
    ASSERT_TRUE(ctx != nullptr);

    ctx->global_scale(std::pow(2, 40));
    auto sk = ctx->secret_key();

    auto l = CKKSVector::Create(ctx, std::vector<double>({1, 2, 3}));
    auto r = CKKSVector::Create(ctx, std::vector<double>({3, 4, 4}));

    auto buffer = l->save();
    auto newl = CKKSVector::Create(buffer);

    auto cpy = newl->deepcopy();
    cpy->link_tenseal_context(ctx);
    auto res = cpy->add(r);
    auto ds = res->decryption_share(ctx,sk);
    auto decr = res->add_share(ds)->mk_decode();
    ASSERT_TRUE(are_close(decr.data(), {4, 6, 7}));
}

}  // namespace
}  // namespace tenseal
