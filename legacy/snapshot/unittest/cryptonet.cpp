// #undef ENABLE_EXTENSION

#include <chrono>
#include "extension/BootContext.h"
#include "Testbed.h"

static constexpr int warm_up = 5;

std::vector<Complex> rotate(const std::vector<Complex>& input, int dist, int num_slots) {
    std::vector<Complex> result(num_slots);
    // 处理负数移位或大于 num_slots 的情况
    int shift = (dist % num_slots + num_slots) % num_slots;
    for (int i = 0; i < num_slots; ++i) {
        result[i] = input[(i + shift) % num_slots];
    }
    return result;
}

// 模拟元素级加法
std::vector<Complex> add(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    std::vector<Complex> res = a;
    for (size_t i = 0; i < a.size(); ++i) res[i] += b[i];
    return res;
}

// 模拟元素级乘法
std::vector<Complex> multiply(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    std::vector<Complex> res = a;
    for (size_t i = 0; i < a.size(); ++i) res[i] *= b[i];
    return res;
}

bool Validate_output_data_absolute_error(double *result, double *expect, int len) {
  bool print_all = true;
  // const char* print_all_str = getenv("PRINT_ALL");
  // if(print_all_str != NULL) {
  //   printf("Value of PRINT_ALL: %s\n", print_all_str);
  //   print_all = true;
  // }

  // const char* absolute_error_str = getenv("ABS_ERROR");
  double absolute_error = 0.0001;
  // if(absolute_error_str != NULL) {
  //   printf("Value of ABS_ERROR: %s\n", absolute_error_str);
  //   absolute_error = atof(absolute_error_str);
  // }
  printf("expect absolute error less than: %f\n", absolute_error);
  int count = 0;
  for(int i = 0; i < len; i++) {
    double result_absolute_error = fabs(result[i] - expect[i]);
    if(print_all) {
      printf("index: %d, result: %f, expect: %f, result absolute error=%f, ", i, result[i], expect[i], result_absolute_error);
      if(result_absolute_error > absolute_error) {
        count++;
        printf("%d failed\n", count);
      } else {
        printf("ok\n");
      }
    } else {
      if(result_absolute_error > absolute_error) {
        printf("index: %d, value: %f != %f, result absolute error=%f\n", i, result[i], expect[i], result_absolute_error);
        return false;
      }
    }
  }
  if(print_all && (count != 0)) {
    return false;
  }
  return true;
}

bool Validate_output_data_relative_error(double *result, double *expect, int len) {
  bool print_all = false;
  // const char* print_all_str = getenv("PRINT_ALL");
  // if(print_all_str != NULL) {
  //   printf("Value of PRINT_ALL: %s\n", print_all_str);
  //   print_all = true;
  // }

  // const char* relative_error_str = getenv("REL_ERROR");
  double relative_error = 0.001;
  // if(relative_error_str != NULL) {
  //   printf("Value of REL_ERROR: %s\n", relative_error_str);
  //   relative_error = atof(relative_error_str);
  // }
  printf("expect relative error less than: %f\n", relative_error);
  int count = 0;
  for(int i = 0; i < len; i++) {
    double result_relative_error = fabs(result[i] - expect[i])/expect[i];
    if(print_all) {
      printf("index: %d, result: %f, expect: %f, result relative error=%f, ", i, result[i], expect[i], result_relative_error);
      if(result_relative_error > relative_error) {
        count++;
        printf("%d failed\n", count);
      } else {
        printf("ok\n");
      }
    } else {
      if (result_relative_error > relative_error) {
        printf("index: %d, value: %f != %f, result relative error: %f\n", i, result[i], expect[i], result_relative_error);
        return false;
      }
    }
  }
  if(print_all && (count != 0)) {
    return false;
  }
  return true;
}

TEST_P(Testbed, cryptonet) {
    std::cout << "test cryptonet" << std::endl;
    int num_slots = (1 << log_degree_) / 2;
    int rot_dist;
    // input
    std::vector<Complex> input, weights_vec[26], weights_vec1[8];
    GenerateRandomMessage(input);
    Plaintext<word> weights[26], weights1[8];
    for (int i = 0; i < 25; i++) {
      GenerateRandomMessage(weights_vec[i]);
      Encode(weights[i], weights_vec[i], 16);
    }
    GenerateRandomMessage(weights_vec[25]);
    Encode(weights[25], weights_vec[25], 15);
    for (int i = 0; i < 8; i++) {
      GenerateRandomMessage(weights_vec1[i]);
      Encode(weights1[i], weights_vec1[i], 14);
    }
    // coding to plaintext and encrypting to ciphertext
    Ciphertext<word> ct1, tmp33, tmp34, tmp35, dup1, result4, grid6, grid6_1, input_dup_n1_8, result11;
    EncodeAndEncrypt(ct1, input, 16);

    // tmp33 = (ct1 << rot_dist) + ct1
    rot_dist = num_slots - 784;
    interface_->PrepareRotationKey(rot_dist, param_->max_level_);
    context_->HRotAdd(tmp33, ct1, ct1, interface_->GetRotationKey(rot_dist), rot_dist);
    
    // tmp34 = (ct1 << rot_dist) + tmp33
    rot_dist = num_slots - 1568;
    interface_->PrepareRotationKey(rot_dist, param_->max_level_);
    context_->HRotAdd(tmp34, ct1, tmp33, interface_->GetRotationKey(rot_dist), rot_dist);

    // tmp35 = (ct1 << rot_dist) + tmp34
    rot_dist = num_slots - 2352;
    interface_->PrepareRotationKey(rot_dist, param_->max_level_);
    context_->HRotAdd(tmp35, ct1, tmp34, interface_->GetRotationKey(rot_dist), rot_dist);

    // tmp35 = (ct1 << rot_dist) + tmp34
    rot_dist = num_slots - 3136;
    interface_->PrepareRotationKey(rot_dist, param_->max_level_);
    context_->HRotAdd(dup1, ct1, tmp35, interface_->GetRotationKey(rot_dist), rot_dist);

    // block[i] = dup1 << _cst_13[i]
    Ciphertext<word> block[25], input_block_n1_10[8];
    int32_t _cst_13[25] = {
      num_slots-58, num_slots-57, num_slots-56, num_slots-55, num_slots-54, num_slots-30, num_slots-29, num_slots-28,
      num_slots-27, num_slots-26, num_slots-2, num_slots-1, 0, 1, 2, 26,
      27, 28, 29, 30, 54, 55, 56, 57,
      58
    };
    for (int i = 0; i < 25; i = i + 1) {
      if(_cst_13[i] == 0){
        context_->Copy(block[i], dup1);
        continue;
      }
      interface_->PrepareRotationKey(_cst_13[i], param_->max_level_);
      context_->HRot(block[i], dup1, interface_->GetRotationKey(_cst_13[i]), _cst_13[i]);
    }

    for (int i = 0; i < 1; i = i + 1) {
      for (int j = 0; j < 25; j = j + 1) {
        Ciphertext<word> tmp;
        // Pt_from_msg_ofst(&_pgen_tmp_3_36, 0/* cst_11 */, ((cap_block_n1_7 + (num_grid_n1_5 * 25))) * (3920), 3920, 1, 7);
        context_->Mult(tmp, block[j], weights[j]);
        if(j == 0){
          context_->Copy(grid6, tmp);
        } else {
          context_->Add(grid6, grid6, tmp);
        }
      }
      context_->Rescale(grid6_1, grid6);
      rot_dist = i * 784;
      if(rot_dist != 0) {
        interface_->PrepareRotationKey(rot_dist, param_->max_level_);
        context_->HRotAdd(result4, grid6_1, result4, interface_->GetRotationKey(rot_dist), rot_dist);
      }else{
        context_->Copy(result4, grid6_1);
      }
    }
    
    // Pt_from_msg(&_pgen_tmp_5_38, 1 /* cst_12 */, 3920, 1, 6);
    context_->Add(result4, result4, weights[25]);
    Ciphertext<word> _preg_268435457;
    context_->Copy(_preg_268435457, result4);
    context_->HMult(input_dup_n1_8, _preg_268435457, _preg_268435457, interface_->GetMultiplicationKey(), true);
    // context_->Mult(dup8, result4, result4);
    // context_->Rescale(dup8, dup8);
    // context_->Relinearize(dup8, dup8, interface_->GetMultiplicationKey());
    int32_t _cst_16[8] = {
      0, 1, 2, 3, 4, 5, 6, 7
    };
    for (int i = 0; i < 8; i = i + 1) {
      if(_cst_16[i] == 0){
        context_->Copy(input_block_n1_10[i], input_dup_n1_8);
        continue;
      }
      interface_->PrepareRotationKey(_cst_16[i], param_->max_level_);
      context_->HRot(input_block_n1_10[i], input_dup_n1_8, interface_->GetRotationKey(_cst_16[i]), _cst_16[i]);
    }
    Ciphertext<word> gemm_result_n2_11;
    for (int gs_n2_12 = 0; gs_n2_12 < 16; gs_n2_12 = gs_n2_12 + 1) {
      Ciphertext<word> block_result_13;
      for (int bs_n2_14 = 0; bs_n2_14 < 8; bs_n2_14 = bs_n2_14 + 1) {
        Ciphertext<word> tmp;
        // Pt_from_msg_ofst(&_pgen_tmp_10_43, 2/* cst_15 */, ((bs_n2_14 + (gs_n2_12 * 8))) * (4096), 4096, 1, 5);
        context_->Mult(tmp, input_block_n1_10[bs_n2_14], weights1[bs_n2_14]);
        if(bs_n2_14 == 0){
          context_->Copy(block_result_13, tmp);
        } else {
          context_->Add(block_result_13, block_result_13, tmp);
        }
      }
      rot_dist = gs_n2_12 * 8;
      if(rot_dist != 0) {
        interface_->PrepareRotationKey(rot_dist, param_->max_level_);
        context_->HRotAdd(gemm_result_n2_11, block_result_13, gemm_result_n2_11, interface_->GetRotationKey(rot_dist), rot_dist);
      }else{
        context_->Copy(gemm_result_n2_11, block_result_13);
      }
    }
    Ciphertext<word> reduce_intra_15;
    context_->Rescale(reduce_intra_15, gemm_result_n2_11);
    for (int ri_n2_16 = 0; ri_n2_16 < 5; ri_n2_16 = ri_n2_16 + 1) {
      rot_dist = (1 << ri_n2_16) * 128;
      interface_->PrepareRotationKey(rot_dist, param_->max_level_);
      Ciphertext<word> tmp;
      context_->HRot(tmp, reduce_intra_15, interface_->GetRotationKey(rot_dist), rot_dist);
      context_->Add(reduce_intra_15, tmp, reduce_intra_15);
    }

    std::vector<Complex> tmp45, tmp47;
    GenerateRandomMessage(tmp45);
    Plaintext<word> tmp45_pt;
    Encode(tmp45_pt, tmp45, 13);
    context_->Add(reduce_intra_15, reduce_intra_15, tmp45_pt);

    GenerateRandomMessage(tmp47);
    Plaintext<word> tmp47_pt;
    Encode(tmp47_pt, tmp47, 13);
    Ciphertext<word> tmp46, p461, input_dup_n2_17, input_block_n2_19[2];
    context_->Mult(tmp46, reduce_intra_15, tmp47_pt);
    Ciphertext<word> p331 ,p462,p463;
    context_->Rescale(p331, tmp46);

    // std::shared_ptr<BootContext<word>> boot_context =
    //   std::dynamic_pointer_cast<BootContext<word>>(context_);
    // boot_context->PrepareEvalMod();
    // boot_context->PrepareEvalSpecialFFT(num_slots);
    // EvkRequest req;
    // boot_context->AddRequiredRotations(req, num_slots);
    // interface_->PrepareRotationKey(req);
    // Ciphertext<word> ct_res;
    // boot_context->Boot(p462, p331, interface_->GetEvkMap());

    context_->HMult(p461, p331, p331, interface_->GetMultiplicationKey(), true);
    // context_->Mult(p461, p331, p331_pt);
    // context_->Rescale(p462, p461);
    // context_->Relinearize(p463, p462, interface_->GetMultiplicationKey());



    // rot_dist = num_slots - 100;
    // interface_->PrepareRotationKey(rot_dist, param_->max_level_);
    // context_->HRotAdd(input_dup_n2_17, p461, p461, interface_->GetRotationKey(rot_dist), rot_dist);
    
    // int32_t _cst_19[2] = {
    //   0, 1
    // };
    // for (int bs_input_n2_18 = 0; bs_input_n2_18 < 2; bs_input_n2_18 = bs_input_n2_18 + 1) {
    //   if(_cst_19[bs_input_n2_18] == 0){
    //     context_->Copy(input_block_n2_19[bs_input_n2_18], input_dup_n2_17);
    //     continue;
    //   }
    //   interface_->PrepareRotationKey(_cst_19[bs_input_n2_18], param_->max_level_);
    //   context_->HRot(input_block_n2_19[bs_input_n2_18], input_dup_n2_17, interface_->GetRotationKey(_cst_19[bs_input_n2_18]), _cst_19[bs_input_n2_18]);
    // }

    // std::vector<Complex> weights_vec2[2];
    // Plaintext<word> weights2[2];
    // for(int i=0; i<2; i++){
    //   GenerateRandomMessage(weights_vec2[i]);
    //   Encode(weights2[i], weights_vec2[i], 2);
    // }
    // Ciphertext<word> gemm_result_n3_20;
    // for (int gs_n3_21 = 0; gs_n3_21 < 5; gs_n3_21 = gs_n3_21 + 1) {
    //   Ciphertext<word> block_result_22, tmp1;
    //   for (int bs_n3_23 = 0; bs_n3_23 < 2; bs_n3_23 = bs_n3_23 + 1) {
    //     // block19 = input_block_n2_19[bs_n3_23];
    //     Ciphertext<word> tmp;
    //     // Pt_from_msg_ofst(&_pgen_tmp_19_52, 5/* cst_18 */, ((bs_n3_23 + (gs_n3_21 * 2))) * (110), 110, 1, 2);
    //     context_->Mult(tmp, input_block_n2_19[bs_n3_23], weights2[bs_n3_23]);
    //     if(bs_n3_23 == 0){
    //       context_->Copy(tmp1, tmp);
    //     } else {
    //       context_->Add(tmp1, tmp1, tmp);
    //     }
    //   }
    //   context_->Rescale(block_result_22, tmp1);
    //   // level = 1     
    //   rot_dist = gs_n3_21 * 2;
    //   if(rot_dist != 0) {
    //     interface_->PrepareRotationKey(rot_dist, param_->max_level_);
    //     context_->HRotAdd(gemm_result_n3_20, block_result_22, gemm_result_n3_20, interface_->GetRotationKey(rot_dist), rot_dist);
    //   }else{
    //     context_->Copy(gemm_result_n3_20, block_result_22);
    //   }
    // }

    // Ciphertext<word> reduce_intra_24;
    // context_->Copy(reduce_intra_24, gemm_result_n3_20);
    // for (int ri_n3_25 = 1; ri_n3_25 < 10; ri_n3_25 = ri_n3_25 + 1) {
    //   rot_dist = ri_n3_25 * 10;
    //   interface_->PrepareRotationKey(rot_dist, param_->max_level_);
    //   context_->HRotAdd(reduce_intra_24, gemm_result_n3_20, reduce_intra_24, interface_->GetRotationKey(rot_dist), rot_dist);
    // }

    // std::vector<Complex> tmp54;
    // GenerateRandomMessage(tmp54);
    // Plaintext<word> tmp54_pt;
    // Encode(tmp54_pt, tmp54, 1);
    // // Pt_from_msg(&_pgen_tmp_21_54, 6 /* cst_5 */, 10, 1, 1);
    // context_->Add(reduce_intra_24, reduce_intra_24, tmp54_pt);

    std::vector<Complex> cres;
    DecryptAndDecode(cres, p461);

    //plaintext compute
    std::vector<Complex> dup1_, block_[25];
    for (int i = 0; i < static_cast<int>(input.size()); i++) {
      dup1_.push_back(
        input[i]
        + input[(i + num_slots - 784) % num_slots]
        + input[(i + num_slots - 1568) % num_slots]
        + input[(i + num_slots - 2352) % num_slots]
        + input[(i + num_slots - 3136) % num_slots]
      );
    }
    for (int i = 0; i < 25; i++) {
      block_[i] = rotate(dup1_, _cst_13[i], num_slots);
    }
    std::vector<Complex> grid6_(num_slots, {0, 0});
    for (int j = 0; j < 25; j++) {
        std::vector<Complex> tmp_prod = multiply(block_[j], weights_vec[j]);
        grid6_ = add(grid6_, tmp_prod);
    }
    std::vector<Complex> result4_ = add(grid6_, weights_vec[25]);
    std::vector<Complex> input_dup_n1_8_ = multiply(result4_, result4_);

    std::vector<Complex> input_block_n1_10_[8];
    for (int i = 0; i < 8; i++) {
      input_block_n1_10_[i] = rotate(input_dup_n1_8_, _cst_16[i], num_slots);
    }
    std::vector<Complex> gemm_result_n2_11_(num_slots, {0, 0});
    for (int gs_n2_12 = 0; gs_n2_12 < 16; gs_n2_12 = gs_n2_12 + 1) {
        std::vector<Complex> block_result_13(num_slots, {0, 0});
        for (int bs_n2_14 = 0; bs_n2_14 < 8; bs_n2_14 = bs_n2_14 + 1) {
            std::vector<Complex> tmp = multiply(input_block_n1_10_[bs_n2_14], weights_vec1[bs_n2_14]);
            block_result_13 = add(block_result_13, tmp);
        }
        int current_rot_dist = gs_n2_12 * 8;
        if (current_rot_dist != 0) {
            std::vector<Complex> rotated_block = rotate(block_result_13, current_rot_dist, num_slots);
            gemm_result_n2_11_ = add(gemm_result_n2_11_, rotated_block);
        } else {
            gemm_result_n2_11_ = block_result_13;
        }
    }

    std::vector<Complex> reduce_intra_15_(gemm_result_n2_11_);
    for (int ri_n2_16 = 0; ri_n2_16 < 5; ri_n2_16 = ri_n2_16 + 1) {
      int current_rot_dist = (1 << ri_n2_16) * 128;
      reduce_intra_15_ = add(reduce_intra_15_, rotate(reduce_intra_15_, current_rot_dist, num_slots));
    }
    reduce_intra_15_ = add(reduce_intra_15_, tmp45);
    reduce_intra_15_ = multiply(reduce_intra_15_, tmp47);
    
    std::vector<Complex> p461_ = multiply(reduce_intra_15_, reduce_intra_15_);

    // std::vector<Complex> input_dup_n2_17_(num_slots);
    // int rot_dist_init = num_slots - 100;
    // std::vector<Complex> rotated_p461 = rotate(p461_, rot_dist_init, num_slots);
    // input_dup_n2_17_ = add(p461_, rotated_p461);


    // std::vector<Complex> input_block_n2_19_[2];
    // for (int i = 0; i < 2; i++) {
    //     input_block_n2_19_[i] = rotate(input_dup_n2_17_, _cst_19[i], num_slots);
    // }

    // std::vector<Complex> gemm_result_n3_20_(num_slots, {0, 0});
    // for (int gs_n3_21 = 0; gs_n3_21 < 5; gs_n3_21++) {
    //   std::vector<Complex> block_result_22_(num_slots, {0, 0});
    //   for (int bs_n3_23 = 0; bs_n3_23 < 2; bs_n3_23++) {
    //     std::vector<Complex> tmp_prod = multiply(input_block_n2_19_[bs_n3_23], weights_vec2[bs_n3_23]);
    //     block_result_22_ = add(block_result_22_, tmp_prod);
    //   }
  
    //   int current_rot_dist = gs_n3_21 * 2;
    //   if (current_rot_dist != 0) {
    //     std::vector<Complex> rotated_block = rotate(block_result_22_, current_rot_dist, num_slots);
    //     gemm_result_n3_20_ = add(gemm_result_n3_20_, rotated_block);
    //   } else {
    //     gemm_result_n3_20_ = block_result_22_;
    //   }
    // }

    // std::vector<Complex> reduce_intra_24_ = gemm_result_n3_20_;
    // for (int ri_n3_25 = 1; ri_n3_25 < 10; ri_n3_25++) {
    //   int stride_dist = ri_n3_25 * 10;
    //   std::vector<Complex> rotated_gemm = rotate(gemm_result_n3_20_, stride_dist, num_slots);
    //   reduce_intra_24_ = add(reduce_intra_24_, rotated_gemm);
    // }

    // reduce_intra_24_ = add(reduce_intra_24_, tmp54);
    CompareMessages(cres, p461_, 1, 1e-3);
    // CompareMessages_Relative(cres, p461_, 1, 1e-3);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed,
    testing::Values("bootparam_40.json"),
    [](const testing::TestParamInfo<Testbed::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    }
);
