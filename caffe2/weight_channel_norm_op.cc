// ------------------------------------------------------------------
// WeightChannelNorm op in Caffe2 for CPU
// Written by Kai Hu
// This is a stand-alone op: Y = gamma * (X - weighted_mu) / weighted_sig + beta
// where
// weighted_mu_ni = sum(mu_nc * weight_ci, c)
// weighted_sig_ni = sum((sig_nc^2 + mu_nc^2) * weight_ci, c) - (sum(mu_nc * weight_ci, c))^2
// ------------------------------------------------------------------

#include "weight_channel_norm_op.h"

#include <array>

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

	namespace {

		template <typename T>
		inline T Cube(const T& x) {
			return x * x * x;
		}

		// for now use softmax outside this op. 
		// TODO: Kai Hu optimum time and space complexity just use sum-one op in this op, the non-linear op should add before this op, use exp for softmax, but suggest use relu.
		template <typename T>
		void SumoneWeight(const int C, T* weight) {
			for (int col = 0; col < C; col++) {
				T sum = T(0);
				for (int row = 0; row < C; row++)
					sum += weight[row * C + col];
				for (int row = 0; row < C; row++)
					weight[row * C + col] /= sum;
			}
		}

		template <typename T, StorageOrder kOrder>
		void WeightChannelMeanVariance(
			const std::array<int, 3>& dims,
			const T* mu,
			const T* var,
			const T* weight,
			T* weighted_mu,
			T* weighted_var) {
			const int C = kOrder == StorageOrder::NCHW ? dims[1] : dims[2];
			const int size = dims[0] * C * C;
			for (int i = 0; i < dims[0] * C; ++i) {  // firstly set to all-zero
				weighted_mu[i] = 0;
				weighted_var[i] = 0;
			}
			for (int i = 0; i < size; ++i) {
				// weighted_mu_nc = sum(mu_nw * weight_wc, w)
				// const int i_n = i / (C * C);
				//const int i_w = i % C;
				//const int i_c = (i / C) % C;
				const int i_nc = i / (C * C) * C + (i / C) % C;
				const int i_wc = (i % C) * C + (i / C) % C;
				const int i_nw = i / (C * C) * C + i % C;
				weighted_mu[i_nc] += mu[i_nw] * weight[i_wc];
			}
			for (int i = 0; i < size; ++i) {
				// weighted_var_nc = sum((var_nw + mu_nw^2 - weighted_mu_nc^2) * weight_wc, w)
				const int i_nc = i / (C * C) * C + (i / C) % C;
				const int i_wc = (i % C) * C + (i / C) % C;
				const int i_nw = i / (C * C) * C + i % C;
				weighted_var[i_nc] += weight[i_wc] * (var[i_nw] + mu[i_nw] * mu[i_nw] - weighted_mu[i_nc] * weighted_mu[i_nc]);
			}
		}

		template <typename T, StorageOrder kOrder>
		void WeightChannelNormForward(
			const std::array<int, 3>& dims,
			const T* X,
			const T* weighted_mu,
			const T* weighted_rsig,
			const T* gamma,
			const T* beta,
			T* Y) {
			constexpr int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
			const int size = dims[0] * dims[1] * dims[2];
			std::array<int, 3> index = { 0, 0, 0 };
			for (int i = 0; i < size; ++i) {
				const int i_mu = index[0] * dims[kCDim] + index[kCDim];
				const int i_gamma = index[kCDim];
				Y[i] = gamma[i_gamma] * (X[i] - weighted_mu[i_mu]) * weighted_rsig[i_mu] + beta[i_gamma];
				math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
			}
		}

		template <typename T, StorageOrder kOrder>
		void ComputeInternalGradients(
			const std::array<int, 3>& dims,
			const T* dY,
			const T* X,
			const T* gamma,
			T* ds,
			T* db) {
			constexpr int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;

			const int size = dims[0] * dims[1] * dims[2];
			std::array<int, 3> index = { 0, 0, 0 };
			for (int i = 0; i < size; ++i) {
				const int i_mu = index[0] * dims[kCDim] + index[kCDim];
				const int i_gamma = index[kCDim];
				ds[i_mu] += gamma[i_gamma] * dY[i] * X[i];
				db[i_mu] += gamma[i_gamma] * dY[i];
				math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
			}
		}

		// Math:
		// Y = gamma * (X - weighted_mu) * weighted_rsig + beta
		// let s = gamma * weighted_rsig
		// let b = beta - gamma * weighted_mu * weighted_rsig
		// Y = s * X + b
		// let n = HxW
		// dL/dX = dL/dY * dY/dX = dL/dY * (s + X * ds/dX + db/dX)
		// ds/dX = gamma * dweighted_rsig/dX
		// db/dX = -gamma * weighted_mu * dweighted_rsig/dX - gamma * weighted_rsig * dweighted_mu/dX
		// Attention: dL/ds, dL/db has timed the gamma, so wo don't time it there in the code.
		// dweighted_rsig/dX = -0.5 * weighted_rsig^3 * dweighted_var/dX
		// dweighted_var/dX = dweighted_var/dvar * dvar/dX + dweighted_var/dmu * dmu/dX
		// dweighted_mu/dX = weight * dmu/dX
		// dsig/dX = 2 * (X - mu) / n * (1 - 1/n)  attention: GN has ignored the (1 - 1/n) reasonable, we also ignore it.
		// dmu/dX = 1 / n
		// dL/dgamma = dL/dY * (dY/ds * ds/dgamma + dY/db * db/dgamma)
		//           = dL/dY * (X - weighted_mu) * weighted_rsig
		// dL/dbeta = dL/dY
		// dL/dweight = dL/dY * (dY/ds * ds/dweighted_sig * dweighted_sig/dweight + dY/db * db/dweighted_mu * dweighted_mu/dweight + dY/db * db/dsig_ * dsig_/dweight)
		//            = dL/dY * (-0.5 * X * rsig_^3 * dsig_/dweight - gamma * rsig_ * dmu_/dweight + gamma * mu_ * rsig_^3 * drsig_/dweight)
		// drsig_/dweight = sig + mu^2 - 2 * Sum(W * mu) * mu
		// dmu_/dweight = mu
		template <typename T, StorageOrder kOrder>
		void WeightChannelNormBackward(
			const std::array<int, 3>& dims,
			const T* dY,
			const T* X,
			const T* mu,
			const T* var,
			const T* weighted_mu,
			const T* weighted_rsig,
			const T* gamma,
			const T* weight,
			const T* ds,
			const T* db,
			T* dX,
			T* dgamma,
			T* dbeta,
			T* dweight) {
			constexpr int kCDim = kOrder == StorageOrder::NCHW ? 1 : 2;
			const int size = dims[0] * dims[1] * dims[2];
			const int HxW = kOrder == StorageOrder::NCHW ? dims[2] : dims[1];
			const int C = kOrder == StorageOrder::NCHW ? dims[1] : dims[2];
			const T denom = T(1) / static_cast<T>(HxW); // 1/n
			std::array<int, 3> index = { 0, 0, 0 };
			for (int i = 0; i < size; ++i) {
				const int i_mu = index[0] * C + index[kCDim]; // (n,c)
				const int i_gamma = index[kCDim]; // (c)
												  // Attention: different channels' mean and variance also affect the X
				T u = T(0);
				T v = T(0);
				for (int j = 0; j < C; ++j) {
					const int i_nj = index[0] * C + j; // (n,c')
					const int i_cj = i_gamma * C + j; // (c,c')
					u += (db[i_nj] * weighted_mu[i_nj] - ds[i_nj]) * weight[i_cj] * (X[i] - weighted_mu[i_nj]) * Cube(weighted_rsig[i_nj]);
					v += db[i_nj] * weighted_rsig[i_nj] * weight[i_cj];
				}
				dX[i] = gamma[i_gamma] * dY[i] * weighted_rsig[i_mu] + (u - v) * denom;
				dgamma[i_gamma] += dY[i] * (X[i] - weighted_mu[i_mu]) * weighted_rsig[i_mu];
				dbeta[i_gamma] += dY[i];
				math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
			}
			for (int i_ck = 0; i_ck < C * C; i_ck++) {
				T dw_val = T(0);
				for (int i_n = 0; i_n < dims[0]; i_n++) {
					const int i_c = i_ck / C;
					const int i_nc = i_n * C + i_ck / C;
					const int i_nk = i_n * C + i_ck % C;
					const T u =
						0.5 * (db[i_nk] * weighted_mu[i_nk] - ds[i_nk]) * Cube(weighted_rsig[i_nk]) * (var[i_nc] + mu[i_nc] * mu[i_nc] - 2 * weighted_mu[i_nk] * mu[i_nc]);
					const T v =
						db[i_nk] * weighted_rsig[i_nk] * mu[i_nc];
					dw_val += u - v;
				}
				dweight[i_ck] = dw_val;
			}
		}

	} // namespace

	template <typename T, class Context>
	bool WeightChannelNormOp<T, Context>::RunOnDeviceImpl(
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
		T* weighted_rsig_data) {
		const std::array<int, 3> dims = order_ == StorageOrder::NCHW
			? std::array<int, 3>{N, C, HxW}
		: std::array<int, 3>{N, HxW, C};
		const std::array<int, 1> axes = order_ == StorageOrder::NCHW
			? std::array<int, 1>{2}
		: std::array<int, 1>{1};

		// Computes mean and variance.
		math::Moments<T, Context>(
			3, dims.data(), 1, axes.data(), X_data, mu_data, var_data, &context_);

		// Computes sum-one weight.
		// SumoneWeight<T>(C, weight_data);
		// Computes weighted mean and variance.
		if (order_ == StorageOrder::NCHW) {
			WeightChannelMeanVariance<T, StorageOrder::NCHW>(
				dims, mu_data, var_data, weight_data, weighted_mu_data, weighted_rsig_data
				);
		}
		else {
			WeightChannelMeanVariance<T, StorageOrder::NHWC>(
				dims, mu_data, var_data, weight_data, weighted_mu_data, weighted_rsig_data
				);
		}
		// Uses rsqrt to computes 1 / std which is much faster than computes std.
		EigenArrayMap<T>(weighted_rsig_data, C, N) += epsilon_;
		math::Rsqrt<T, CPUContext>(N * C, weighted_rsig_data, weighted_rsig_data, &context_);

		// Computes Y = gamma * (X - weighted_mu) * weighted_rsig + beta.
		if (order_ == StorageOrder::NCHW) {
			WeightChannelNormForward<T, StorageOrder::NCHW>(
				dims, X_data, weighted_mu_data, weighted_rsig_data, gamma_data, beta_data, Y_data);
		}
		else {
			WeightChannelNormForward<T, StorageOrder::NHWC>(
				dims, X_data, weighted_mu_data, weighted_rsig_data, gamma_data, beta_data, Y_data);
		}
		return true;
	}

	// Math:
	// let: s = gamma * rsig
	// let: b = beta - mu * gamma * rsig
	// then: Y = s * X + b
	template <typename T, class Context>
	bool WeightChannelNormGradientOp<T, Context>::RunOnDeviceImpl(
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
		T* dweight_data) {
		const std::array<int, 3> dims = order_ == StorageOrder::NCHW
			? std::array<int, 3>{N, C, HxW}
		: std::array<int, 3>{N, HxW, C};

		// Computes dL/ds and dL/db.
		// dL/ds = Sum(dL/dY * gamma * X)
		// dL/db = Sum(dL/dY * gamma)
		// Attention: dL/ds, dL/db has timed the gamma to accelerate the computation.
		ds_.Resize(N, C);
		db_.Resize(N, C);
		T* ds_data = ds_.template mutable_data<T>();
		T* db_data = db_.template mutable_data<T>();
		math::Set<T, Context>(N * C, T(0), ds_data, &context_);
		math::Set<T, Context>(N * C, T(0), db_data, &context_);
		if (order_ == StorageOrder::NCHW) {
			ComputeInternalGradients<T, StorageOrder::NCHW>(
				dims, dY_data, X_data, gamma_data, ds_data, db_data);
		}
		else {
			ComputeInternalGradients<T, StorageOrder::NHWC>(
				dims, dY_data, X_data, gamma_data, ds_data, db_data);
		}

		// Computes dL/dX, dL/dgamma, dL/dbeta and dL/dweight.
		math::Set<T, Context>(C, T(0), dgamma_data, &context_);
		math::Set<T, Context>(C, T(0), dbeta_data, &context_);
		math::Set<T, Context>(C * C, T(0), dweight_data, &context_);
		if (order_ == StorageOrder::NCHW) {
			WeightChannelNormBackward<T, StorageOrder::NCHW>(
				dims,
				dY_data,
				X_data,
				mu_data,
				var_data,
				weighted_mu_data,
				weighted_rsig_data,
				gamma_data,
				weight_data,
				ds_data,
				db_data,
				dX_data,
				dgamma_data,
				dbeta_data,
				dweight_data);
		}
		else {
			WeightChannelNormBackward<T, StorageOrder::NHWC>(
				dims,
				dY_data,
				X_data,
				mu_data,
				var_data,
				weighted_mu_data,
				weighted_rsig_data,
				gamma_data,
				weight_data,
				ds_data,
				db_data,
				dX_data,
				dgamma_data,
				dbeta_data,
				dweight_data);
		}
		return true;
	}

	REGISTER_CPU_OPERATOR(WeightChannelNorm, WeightChannelNormOp<float, CPUContext>);
	REGISTER_CPU_OPERATOR(
		WeightChannelNormGradient,
		WeightChannelNormGradientOp<float, CPUContext>);

	// Warning: mu, var, weighted_mu and weighted_rsig are for backward usage or reference. They should NOT be
	// used as forward activations as they have no direct gradients computed.

	// Input: X, gamma, beta, weight; Output: Y, mu, var, weighted_mu, weighted_rsig
	OPERATOR_SCHEMA(WeightChannelNorm)
		.NumInputs(4)
		.NumOutputs(5)
		.SetDoc(R"DOC(
		Weight Channel Normalization (WCN) operation
		)DOC")
		.Arg("epsilon", "(float) default 1e-5; small constant added to var.")
		.Input(
			0,
			"X",
			">=4D feature map input of shape (N, C, H, W) or (N, C, T, H, W)")
		.Input(
			1,
			"gamma",
			"The scale as a 1-dimensional tensor of size C to be applied to the "
			"output.")
		.Input(
			2,
			"beta",
			"The bias as a 1-dimensional tensor of size C to be applied to the "
			"output.")
		.Input(
			3,
			"weight",
			"The weight as a 2-dimensional tensor of size (C,C) to be applied to"
			"weight the mean and variance."
		)
		.Output(0, "Y", "The output >=4-dimensional tensor of the same shape as X.")
		.Output(
			1,
			"mean",
			"The mean of shape (N, C). "
			"For backward usage or reference. "
			"Cannot be used as activations.")
		.Output(
			2,
			"var",
			"The variance of shape (N, C). "
			"For backward usage or reference. "
			"Cannot be used as activations.")
		.Output(
			3,
			"weighted_mean",
			"The weighted mean of shape (N, C). Weighted by Input weight. "
			"For backward usage or reference. "
			"Cannot be used as activations.")
		.Output(
			4,
			"weighted_rsig",
			"The weighted inv_std of shape (N, C). Weighted by Input weight. "
			"For backward usage or reference. "
			"Cannot be used as activations.");

	// Input: dY, X, gamma, beta, weight, mu, var, weighted_mu, weighted_rsig; Output: dX, dgamma, dbeta, dweight
	OPERATOR_SCHEMA(WeightChannelNormGradient).NumInputs(9).NumOutputs(4);

	class GetWeightChannelNormGradient : public GradientMakerBase {
		using GradientMakerBase::GradientMakerBase;
		vector<OperatorDef> GetGradientDefs() override {
			return SingleGradientDef(
				"WeightChannelNormGradient",
				"",
				vector<string>{GO(0), I(0), I(1), I(2), I(3), O(1), O(2), O(3), O(4)},
				vector<string>{GI(0), GI(1), GI(2), GI(3)});
		}
	};

	REGISTER_GRADIENT(WeightChannelNorm, GetWeightChannelNormGradient);

} // namespace caffe2
