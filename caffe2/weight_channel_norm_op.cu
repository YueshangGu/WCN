// ------------------------------------------------------------------
// WeightChannelNorm op in Caffe2 for GPU
// Written by Kai Hu
// This is a stand-alone op: Y = gamma * (X - weighted_mu) * weighted_rsig + beta
// where
// weighted_mu_ni = sum(mu_nc * weight_ci, c)
// weighted_sig_ni = sum((sig_nc^2 + mu_nc^2) * weight_ci, c) - (sum(mu_nc * weight_ci, c))^2
// ------------------------------------------------------------------

#include "weight_channel_norm_op.h"

#include <array>

#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math_utils.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

__global__ void InvStdCUDAKernel(
    const int size,
    const float epsilon,
    const float* var,
    float* rsig) {
  CUDA_1D_KERNEL_LOOP(i, size) {
#if __CUDA_ARCH__ >= 350
    rsig[i] = rsqrtf(__ldg(var + i) + epsilon);
#else
    rsig[i] = rsqrtf(var[i] + epsilon);
#endif
  }
}

template <typename T, StorageOrder kOrder>
__global__ void WeightChannelNormForwardCUDAKernel(
    const int size,
    const int C,
    const int HxW,
    const T* X,
    const T* weighted_mu,
    const T* weighted_rsig,
    const T* gamma,
    const T* beta,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    const int i_mu = kOrder == StorageOrder::NCHW
        ? i / HxW
        : i / (C * HxW) * C + (i % C);
    const int i_gamma = kOrder == StorageOrder::NCHW ? (i / HxW) % C : i % C;
#if __CUDA_ARCH__ >= 350
    Y[i] = __ldg(gamma + i_gamma) * (__ldg(X + i) - __ldg(weighted_mu + i_mu)) *
            __ldg(weighted_rsig + i_mu) +
        __ldg(beta + i_gamma);
#else
    Y[i] = gamma[i_gamma] * (X[i] - weighted_mu[i_mu]) * weighted_rsig[i_mu] + beta[i_gamma];
#endif
  }
}

// add by Kai Hu, add function to compute the weighted mean and variance. 
template <typename T>
__global__ void WeightChannelMeanCUDAKernel(
    const int N,
    const int C,
    const T* mu,
    const T* weight,
    T* weighted_mu) {
    const int outer_size = N * C;
    const int inner_size = C;
    __shared__ typename BlockReduce<T>::TempStorage wmu_storage;
    for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
        T wmu_val = 0;
        for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
            const int i_nc = i;
            const int i_wc = j * C + i % C;
            const int i_nw = (i / C) * C + j;
#if __CUDA_ARCH__ >= 350
            wmu_val += __ldg(mu + i_nw) * __ldg(weight + i_wc);
#else
            wmu_val += mu[i_nw] * weight[i_wc];
#endif
        }
        wmu_val = BlockReduce<T>(wmu_storage).Reduce(wmu_val, cub::Sum());
        if (threadIdx.x == 0) {
            weighted_mu[i] = wmu_val;
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void WeightChannelVarianceCUDAKernel(
    const int N,
    const int C,
    const T* mu,
    const T* var,
    const T* weight,
    T* weighted_mu,
    T* weighted_var) {
    const int outer_size = N * C;
    const int inner_size = C;
    __shared__ typename BlockReduce<T>::TempStorage wvar_storage;
    for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
        T wvar_val = 0;
        for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
            const int i_nc = i;
            const int i_wc = j * C + i % C;
            const int i_nw = (i / C) * C + j;
#if __CUDA_ARCH__ >= 350
            wvar_val += __ldg(weight + i_wc) * (__ldg(var + i_nw) + __ldg(mu + i_nw) * __ldg(mu + i_nw) - 
                    __ldg(weighted_mu + i_nc) * __ldg(weighted_mu + i_nc));
#else
            wvar_val += weight[i_wc] * (var[i_nw] + mu[i_nw] * mu[i_nw] - weighted_mu[i_nc] * weighted_mu[i_nc]);
#endif
        }
        wvar_val = BlockReduce<T>(wvar_storage).Reduce(wvar_val, cub::Sum());
        if (threadIdx.x == 0) {
            weighted_var[i] = wvar_val;
        }
        __syncthreads();
    }
}

template <typename T, StorageOrder kOrder>
__global__ void ComputeInternalGradientsCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* gamma,
    T* ds,
    T* db) {
  const int outer_size = N * C;
  const int inner_size = HxW;
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T ds_val = 0;
    T db_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int i_gamma = i % C;
      const int index = kOrder == StorageOrder::NCHW
          ? i * inner_size + j
          : ((i / C) * HxW + j) * C + i % C;
#if __CUDA_ARCH__ >= 350
      ds_val += __ldg(gamma + i_gamma) * __ldg(dY + index) * __ldg(X + index);
      db_val += __ldg(gamma + i_gamma) * __ldg(dY + index);
#else
      ds_val += gamma[i_gamma] * dY[index] * X[index];
      db_val += gamma[i_gamma] * dY[index];
#endif
    }
    ds_val = BlockReduce<T>(ds_storage).Reduce(ds_val, cub::Sum());
    db_val = BlockReduce<T>(db_storage).Reduce(db_val, cub::Sum());
    if (threadIdx.x == 0) {
      ds[i] = ds_val;
      db[i] = db_val;
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void ComputeTempVariable(
    const int N,
    const int C,
    const int HxW,
    const T* weighted_mu,
    const T* weighted_rsig,
    const T* weight,
    const T* ds,
    const T* db,
    T* temp_u,
    T* temp_v) {
        const T denom = T(1) / static_cast<T>(HxW);
        const int outer_size = N * C;
        const int inner_size = C;
        __shared__ typename BlockReduce<T>::TempStorage u_storage;
        __shared__ typename BlockReduce<T>::TempStorage v_storage;
        for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
            T u_val = 0;
            T v_val = 0;
            for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
                const int i_nj = (i / C) * C + j;
                const int i_cj = (i % C) * C + j;
#if __CUDA_ARCH__ >= 350
              u_val += (__ldg(db + i_nj) * __ldg(weighted_mu + i_nj) - __ldg(ds + i_nj)) * 
                  __ldg(weight + i_cj) * math::utils::Cube<T>(__ldg(weighted_rsig + i_nj));
              v_val += (__ldg(db + i_nj) * __ldg(weighted_mu + i_nj) - __ldg(ds + i_nj)) * 
                  __ldg(weight + i_cj) * math::utils::Cube<T>(__ldg(weighted_rsig + i_nj)) * __ldg(weighted_mu + i_nj);
              v_val += __ldg(db + i_nj) * __ldg(weighted_rsig + i_nj) * __ldg(weight + i_cj);
#else
              u_val += (db[i_nj] * weighted_mu[i_nj] - ds[i_nj]) * weight[i_cj] * math::utils::Cube<T>(weighted_rsig[i_nj]);
              v_val += (db[i_nj] * weighted_mu[i_nj] - ds[i_nj]) * weight[i_cj] * math::utils::Cube<T>(weighted_rsig[i_nj]) * weighted_mu[i_nj];
              v_val += db[i_nj] * weighted_rsig[i_nj] * weight[i_cj];
#endif
            }
            u_val = BlockReduce<T>(u_storage).Reduce(u_val, cub::Sum());
            v_val = BlockReduce<T>(v_storage).Reduce(v_val, cub::Sum());
            if (threadIdx.x == 0) {
                temp_u[i] = u_val * denom;
                temp_v[i] = v_val * denom;
            }
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
// TODO:Kai Hu This function can be optimized
// template <typename T, StorageOrder kOrder>
// __global__ void WeightChannelNormBackwardCUDAKernel(
//     const int size,
//     const int C,
//     const int HxW,
//     const T* dY,
//     const T* X,
//     const T* mu,
//     const T* var,
//     const T* weighted_mu,
//     const T* weighted_rsig,
//     const T* gamma,
//     const T* weight,
//     const T* ds,
//     const T* db,
//     T* dX) {
//   const T denom = T(1) / static_cast<T>(HxW);
//   const int outer_size = size;
//   const int inner_size = C;
//   __shared__ typename BlockReduce<T>::TempStorage u_storage;
//   __shared__ typename BlockReduce<T>::TempStorage v_storage;
//   for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
//     T u_val = 0;
//     T v_val = 0;
//     for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
//       const int i_mu = kOrder == StorageOrder::NCHW
//         ? i / HxW
//         : (i / (C * HxW)) * C + (i % C); // (n,c)
//       const int i_gamma = kOrder == StorageOrder::NCHW ? (i / HxW) % C : i % C; // (c)
//       const int i_nj = i_mu / C * C + j; // (n,c')
//       const int i_cj = i_gamma * C + j; // (c,c')
// #if __CUDA_ARCH__ >= 350
//       u_val += (__ldg(db + i_nj) * __ldg(weighted_mu + i_nj) - __ldg(ds + i_nj)) * 
//           __ldg(weight + i_cj) * (__ldg(X + i) - __ldg(weighted_mu + i_nj)) * 
//           math::utils::Cube<T>(__ldg(weighted_rsig + i_nj));
//       v_val += __ldg(db + i_nj) * __ldg(weighted_rsig + i_nj) * __ldg(weight + i_cj);
// #else
//       u_val += (db[i_nj] * weighted_mu[i_nj] - ds[i_nj]) * weight[i_cj] * (X[i] - weighted_mu[i_nj]) * math::utils::Cube<T>(weighted_rsig[i_nj]);
//       v_val += db[i_nj] * weighted_rsig[i_nj] * weight[i_cj];
// #endif
//     }
//     u_val = BlockReduce<T>(u_storage).Reduce(u_val, cub::Sum());
//     v_val = BlockReduce<T>(v_storage).Reduce(v_val, cub::Sum());
//     if (threadIdx.x == 0) {
//       const int i_mu = kOrder == StorageOrder::NCHW
//         ? i / HxW
//         : (i / (C * HxW)) * C + (i % C); // (n,c)
//       const int i_gamma = kOrder == StorageOrder::NCHW ? (i / HxW) % C : i % C; // (c)
//       dX[i] = gamma[i_gamma] * dY[i] * weighted_rsig[i_mu] + (u_val - v_val) * denom;
//     }
//     __syncthreads();
//   }
// }
template <typename T, StorageOrder kOrder>
__global__ void WeightChannelNormBackwardCUDAKernel(
    const int size,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* weighted_rsig,
    const T* gamma,
    const T* temp_u,
    const T* temp_v,
    T* dX) {
        CUDA_1D_KERNEL_LOOP(i, size) {
            const int i_mu = kOrder == StorageOrder::NCHW 
                ? i / HxW
                : (i / (C * HxW)) * C + (i % C); // (n,c)
            const int i_gamma = kOrder == StorageOrder::NCHW ? (i / HxW) % C : i % C; // (c)
            dX[i] = X[i] * temp_u[i_mu] - temp_v[i_mu] + gamma[i_gamma] * dY[i] * weighted_rsig[i_mu];
        }
}


template <typename T, StorageOrder kOrder>
__global__ void GammaBetaBackwardCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* weighted_mu,
    const T* weighted_rsig,
    T* dgamma,
    T* dbeta) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  __shared__ typename BlockReduce<T>::TempStorage dg_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T dg_val = 0;
    T db_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int n = j / HxW;
      const int index = kOrder == StorageOrder::NCHW
          ? (n * outer_size + i) * HxW + j % HxW
          : j * outer_size + i;
      const int i_mu = n * outer_size + i;
#if __CUDA_ARCH__ >= 350
      dg_val += __ldg(dY + index) * (__ldg(X + index) - __ldg(weighted_mu + i_mu)) *
          __ldg(weighted_rsig + i_mu);
      db_val += __ldg(dY + index);
#else
      dg_val += dY[index] * (X[index] - weighted_mu[i_mu]) * weighted_rsig[i_mu];
      db_val += dY[index];
#endif
    }
    dg_val = BlockReduce<T>(dg_storage).Reduce(dg_val, cub::Sum());
    db_val = BlockReduce<T>(db_storage).Reduce(db_val, cub::Sum());
    if (threadIdx.x == 0) {
      dgamma[i] = dg_val;
      dbeta[i] = db_val;
    }
    __syncthreads();
  }
}

template <typename T, StorageOrder kOrder>
__global__ void WeightBackwardCUDAKernel(
    const int N,
    const int C,
    const T* mu,
    const T* var,
    const T* weighted_mu,
    const T* weighted_rsig,
    const T* gamma,
    const T* ds,
    const T* db,
    T* dweight) {
  const int outer_size = C * C;
  const int inner_size = N;
  __shared__ typename BlockReduce<T>::TempStorage dw_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T dw_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
    //   const int n = j;
    //   const int k = i % C;
    //   const int c = i / C;
      const int i_nc = j * C + i / C;
      const int i_c = i / C;
      const int i_nk = j * C + i % C;
#if __CUDA_ARCH__ >= 350
      const T u = 0.5 * (__ldg(db + i_nk) * __ldg(weighted_mu + i_nk) - __ldg(ds + i_nk)) *
          math::utils::Cube<T>(__ldg(weighted_rsig + i_nk)) * 
          (__ldg(var + i_nc) + __ldg(mu + i_nc) * __ldg(mu + i_nc) - 2 * __ldg(weighted_mu + i_nk) * __ldg(mu + i_nc));
      const T v = __ldg(db + i_nk) * __ldg(weighted_rsig + i_nk) * __ldg(mu + i_nc);
      dw_val += u - v;
#else
      const T u =
		  0.5 * (db[i_nk] * weighted_mu[i_nk] - ds[i_nk]) * math::utils::Cube<T>(weighted_rsig[i_nk]) * (var[i_nc] + mu[i_nc] * mu[i_nc] - 2 * weighted_mu[i_nk] * mu[i_nc]);
      const T v =
          db[i_nk] * weighted_rsig[i_nk] * mu[i_nc];
      dw_val += u - v;
#endif
    }
    dw_val = BlockReduce<T>(dw_storage).Reduce(dw_val, cub::Sum());
    if (threadIdx.x == 0) {
        dweight[i] = dw_val;
      }
      __syncthreads();
  }
}

} // namespace

template <>
bool WeightChannelNormOp<float, CUDAContext>::RunOnDeviceImpl(
    const int N,
    const int C,
    const int HxW,
    const float* X_data,
    const float* gamma_data,
    const float* beta_data,
    const float* weight_data,
    float* Y_data,
    float* mu_data,
    float* var_data,
    float* weighted_mu_data,
    float* weighted_rsig_data) {
  const std::array<int, 3> dims = order_ == StorageOrder::NCHW
      ? std::array<int, 3>{N, C, HxW}
      : std::array<int, 3>{N, HxW, C};
  const std::array<int, 1> axes = order_ == StorageOrder::NCHW
      ? std::array<int, 1>{2}
      : std::array<int, 1>{1};

  // Computes mean and variance.
  math::Moments<float, CUDAContext>(
      3, dims.data(), 1, axes.data(), X_data, mu_data, var_data, &context_);
  // Computes weighted mean and variance.
  WeightChannelMeanCUDAKernel<float>
    <<<std::min(N * C, CAFFE_MAXIMUM_NUM_BLOCKS),
       CAFFE_CUDA_NUM_THREADS,
       0,
       context_.cuda_stream()>>>(
        N,
        C,
        mu_data,
        weight_data,
        weighted_mu_data);
  WeightChannelVarianceCUDAKernel<float>
    <<<std::min(N * C, CAFFE_MAXIMUM_NUM_BLOCKS),
       CAFFE_CUDA_NUM_THREADS,
       0,
       context_.cuda_stream()>>>(
        N,
        C,
        mu_data,
        var_data,
        weight_data,
        weighted_mu_data,
        weighted_rsig_data);
  // Uses rsqrt to computes 1 / std which is much faster than computes std.
  InvStdCUDAKernel<<<
      CAFFE_GET_BLOCKS(N * C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N * C, epsilon_, weighted_rsig_data, weighted_rsig_data);

  // Computes Y = gamma * (X - weighted_mu) * weighted_rsig + beta.
  const int size = N * C * HxW;
  if (order_ == StorageOrder::NCHW) {
    WeightChannelNormForwardCUDAKernel<float, StorageOrder::NCHW>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            size,
            C,
            HxW,
            X_data,
            weighted_mu_data,
            weighted_rsig_data,
            gamma_data,
            beta_data,
            Y_data);
  } else {
    WeightChannelNormForwardCUDAKernel<float, StorageOrder::NHWC>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            size,
            C,
            HxW,
            X_data,
            weighted_mu_data,
            weighted_rsig_data,
            gamma_data,
            beta_data,
            Y_data);
  }
  return true;
}

// Math:
// let: s = gamma * rsig
// let: b = beta - mu * gamma * rsig
// then: Y = s * X + b
template <>
bool WeightChannelNormGradientOp<float, CUDAContext>::RunOnDeviceImpl(
    const int N,
    const int C,
    const int HxW,
    const float* dY_data,
    const float* X_data,
    const float* mu_data,
    const float* var_data,
    const float* weighted_mu_data,
    const float* weighted_rsig_data,
    const float* gamma_data,
    const float* weight_data,
    float* dX_data,
    float* dgamma_data,
    float* dbeta_data,
    float* dweight_data) {
  const int size = N * C * HxW;
  ds_.Resize(N, C);
  db_.Resize(N, C);
  temp_u.Resize(N, C);
  temp_v.Resize(N, C);
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  float* temp_u_data = temp_u.mutable_data<float>();
  float* temp_v_data = temp_v.mutable_data<float>();
  if (order_ == StorageOrder::NCHW) {
    // Computes dL/ds and dL/db.
    // dL/ds = Sum(dL/dY * gamma * X)
    // dL/db = Sum(dL/dY * gamma)
    // Attention: dL/ds, dL/db has timed the gamma to accelerate the computation.
    ComputeInternalGradientsCUDAKernel<float, StorageOrder::NCHW>
        <<<std::min(N * C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N, C, HxW, dY_data, X_data, gamma_data, ds_data, db_data);
    ComputeTempVariable<float>
        <<<std::min(N * C, CAFFE_MAXIMUM_NUM_BLOCKS),
            CAFFE_CUDA_NUM_THREADS,
            0,
            context_.cuda_stream()>>>(
            N, C, HxW, weighted_mu_data, weighted_rsig_data,
            weight_data, ds_data, db_data, temp_u_data, temp_v_data);
    // Computes dL/dX.
    WeightChannelNormBackwardCUDAKernel<float, StorageOrder::NCHW>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            size,
            C,
            HxW,
            dY_data,
            X_data,
            weighted_rsig_data,
            gamma_data,
            temp_u_data,
            temp_v_data,
            dX_data);

    // Computes dL/dgamma and dL/dbeta.
    GammaBetaBackwardCUDAKernel<float, StorageOrder::NCHW>
        <<<std::min(C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            C,
            HxW,
            dY_data,
            X_data,
            weighted_mu_data,
            weighted_rsig_data,
            dgamma_data,
            dbeta_data);
    // Computes dL/dweight
    WeightBackwardCUDAKernel<float, StorageOrder::NCHW>
        <<<std::min(C * C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            C,
            mu_data,
            var_data,
            weighted_mu_data,
            weighted_rsig_data,
            gamma_data,
            ds_data,
            db_data,
            dweight_data);
  } else {
    // Computes dL/ds and dL/db.
    // dL/ds = Sum(dL/dY * gamma * X)
    // dL/db = Sum(dL/dY * gamma)
    ComputeInternalGradientsCUDAKernel<float, StorageOrder::NHWC>
        <<<std::min(N * C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N, C, HxW, dY_data, X_data, gamma_data, ds_data, db_data);
    ComputeTempVariable<float>
        <<<std::min(N * C, CAFFE_MAXIMUM_NUM_BLOCKS),
            CAFFE_CUDA_NUM_THREADS,
            0,
            context_.cuda_stream()>>>(
            N, C, HxW, weighted_mu_data, weighted_rsig_data,
            weight_data, ds_data, db_data, temp_u_data, temp_v_data);    
    // Computes dL/dX.
    WeightChannelNormBackwardCUDAKernel<float, StorageOrder::NHWC>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            size,
            C,
            HxW,
            dY_data,
            X_data,
            weighted_rsig_data,
            gamma_data,
            temp_u_data,
            temp_v_data,
            dX_data);

    // Computes dL/dgamma and dL/dbeta.
    GammaBetaBackwardCUDAKernel<float, StorageOrder::NHWC>
        <<<std::min(C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            C,
            HxW,
            dY_data,
            X_data,
            weighted_mu_data,
            weighted_rsig_data,
            dgamma_data,
            dbeta_data);
    WeightBackwardCUDAKernel<float, StorageOrder::NHWC>
        <<<std::min(C * C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            C,
            mu_data,
            var_data,
            weighted_mu_data,
            weighted_rsig_data,
            gamma_data,
            ds_data,
            db_data,
            dweight_data);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(WeightChannelNorm, WeightChannelNormOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    WeightChannelNormGradient,
    WeightChannelNormGradientOp<float, CUDAContext>);

} // namespace caffe2
