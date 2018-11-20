#ifndef CAFFE2_OPERATORS_WEIGHT_CHANNEL_NORM_OP_H_
#define CAFFE2_OPERATORS_WEIGHT_CHANNEL_NORM_OP_H_

#include <string>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class WeightChannelNormOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  WeightChannelNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<std::string>("order", "NCHW"))) {
    CAFFE_ENFORCE_NE(
        order_,
        StorageOrder::UNKNOWN,
        "order should be either \"NCHW\" or \"NHWC\".");
  }

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& gamma = Input(GAMMA);
    const auto& beta = Input(BETA);
    const auto& weight = Input(WEIGHT);
    const int ndim = X.ndim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const int HxW = X.size() / (N * C);
    CAFFE_ENFORCE_EQ(gamma.size(), C);
    CAFFE_ENFORCE_EQ(beta.size(), C);
    CAFFE_ENFORCE_EQ(weight.size(), C*C);
    auto* Y = Output(OUTPUT);
    auto* mu = Output(MU);
    auto* var = Output(VAR);
    auto* weighted_mu = Output(WEIGHTED_MU);
    auto* weighted_rsig = Output(WEIGHTED_INV_SIG);
    Y->ResizeLike(X);
    mu->Resize(N, C);
    var->Resize(N, C);
    weighted_mu->Resize(N, C);
    weighted_rsig->Resize(N,C);
    return RunOnDeviceImpl(
        N,
        C,
        HxW,
        X.template data<T>(),
        gamma.template data<T>(),
        beta.template data<T>(),
        weight.template data<T>(),
        Y->template mutable_data<T>(),
        mu->template mutable_data<T>(),
        var->template mutable_data<T>(),
        weighted_mu->template mutable_data<T>(),
        weighted_rsig->template mutable_data<T>());
  }

 protected:
  bool RunOnDeviceImpl(
      const int N,
      const int C,
      const int HxW,
      const T* X_data,
      const T* gamma_data,
      const T* beta_data,
      const T* weight_data,
      T* Y_data,
      T* mu_data,
      T* var_data,
      T* weighted_mu_data,
      T* weighted_rsig_data);

  const float epsilon_;
  const StorageOrder order_;

  // Input: X, gamma, beta, weight
  // Output: Y, mu, var, weighted_mu, weight_inv_sig
  INPUT_TAGS(INPUT, GAMMA, BETA, WEIGHT);
  OUTPUT_TAGS(OUTPUT, MU, VAR, WEIGHTED_MU, WEIGHTED_INV_SIG);
};

template <typename T, class Context>
class WeightChannelNormGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  WeightChannelNormGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<std::string>("order", "NCHW"))) {
    CAFFE_ENFORCE_NE(
        order_,
        StorageOrder::UNKNOWN,
        "order should be either \"NCHW\" or \"NHWC\".");
  }

  bool RunOnDevice() override {
    const auto& dY = Input(OUTPUT_GRAD);
    const auto& X = Input(INPUT);
    const auto& gamma = Input(GAMMA);
    const auto& beta = Input(BETA);
    const auto& weight = Input(WEIGHT);
    const auto& mu = Input(MU);
    const auto& var = Input(VAR);
	const auto& weighted_mu = Input(WEIGHTED_MU);
	const auto& weighted_rsig = Input(WEIGHTED_RSIG);
    const int ndim = X.ndim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const int HxW = X.size() / (N * C);
    CAFFE_ENFORCE_EQ(gamma.size(), C);
    CAFFE_ENFORCE_EQ(beta.size(), C);
    CAFFE_ENFORCE_EQ(weight.size(), C * C);
    auto* dX = Output(INPUT_GRAD);
    auto* dgamma = Output(GAMMA_GRAD);
    auto* dbeta = Output(BETA_GRAD);
    auto* dweight = Output(WEIGHT_GRAD);
    dX->ResizeLike(X);
    dgamma->ResizeLike(gamma);
    dbeta->ResizeLike(beta);
    dweight->ResizeLike(weight);
    return RunOnDeviceImpl(
        N,
        C,
        HxW,
        dY.template data<T>(),
        X.template data<T>(),
        mu.template data<T>(),
        var.template data<T>(),
		weighted_mu.template data<T>(),
		weighted_rsig.template data<T>(),
        gamma.template data<T>(),
        weight.template data<T>(),
        dX->template mutable_data<T>(),
        dgamma->template mutable_data<T>(),
        dbeta->template mutable_data<T>(),
        dweight->template mutable_data<T>());
  }

 protected:
  bool RunOnDeviceImpl(
      const int N,
      const int C,
      const int HxW,
      const T* dY_data,
      const T* X_data,
      const T* mu_data,
      const T* var_data,
	  const T* weighted_mu_data,
	  const T* weighted_rsig_data,
      const T* gamma_data,
      const T* weight_data,
      T* dX_data,
      T* dgamma_data,
      T* dbeta_data,
      T* dweight_data);


  const StorageOrder order_;
  
  Tensor ds_{ Context::GetDeviceType() };
  Tensor db_{ Context::GetDeviceType() };
  Tensor temp_u{ Context::GetDeviceType() };
  Tensor temp_v{ Context::GetDeviceType() };
 /* Tensor<Context> ds_;
  Tensor<Context> db_;
  Tensor<Context> temp_u;
  Tensor<Context> temp_v;*/

  // Input: dY, X, gamma, beta, weight, mu, var, weighted_mu, weighted_rsig
  // Output: dX, dgamma, dbeta, dweight
  INPUT_TAGS(OUTPUT_GRAD, INPUT, GAMMA, BETA, WEIGHT, MU, VAR, WEIGHTED_MU, WEIGHTED_RSIG);
  OUTPUT_TAGS(INPUT_GRAD, GAMMA_GRAD, BETA_GRAD, WEIGHT_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_WEIGHT_CHANNEL_NORM_OP_H_
