#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {

// implement output = input @ weight
// input: 1*M,weight:M*K,转置后是K*M,output:1*K
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  // block共享的row：(start_row, end_row)
  // 每个block计算1行
  // THREAD_PER_BLOCK=128, ROW_PER_BLOCK=1
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  // float4 长度
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

  //每次循环对一行进行reduce
#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    // row_offset = p * M
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    // i = tid + k * blockdim.x, 每个block处理blockDim.x * k个数据, x=pack_num/blockDim
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    //剩下的部分, 不是4的倍数
    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    // 对归约到block的部分做最后一次reduce
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();
    // 输出本行的输出
    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}


template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void my_matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M, int K){
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  // input:1*M, weight:K*M, output:K*1
  int start_row = ROW_PER_BLOCK * blockIdx.x;
  int end_row = start_row + ROW_PER_BLOCK;
  if(start_row >= K){
    return;
  }

  // compute float4
  constexpr int pack_size = 4;
  int pack_num = M/pack_size;
  int pack_off = pack_num * pack_size;
  
  // reduce, spmv
#pragma unroll
  for(int p=start_row; p<end_row;++p){
    int row_offset = p * M;
    float4* input_float4 = (float4*)input;
    float4* weight_float4 = (float4*)(weight + row_offset);

    sdata[tid] = 0;
#pragma unroll
    for(int i=tid; i<pack_num; i+=blockDim.x){
      float4 in = input_float4[i];
      float4 wei = weight_float4[i];
      sdata[tid] += in.x * wei.x + in.y * wei.y + in.z * wei.z + in.w * wei.w;
    }

    for(int i=pack_off+tid; i<M; i+=blockDim.x){
      sdata[tid] += input[i] * weight[row_offset + i];
    }
    
    __syncthreads();

    /*
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();*/
    
    // warp reduce
    // tid->sum
    const int warpSize=32;
    constexpr int NUM_WARPS = (THREAD_PER_BLOCK + warpSize - 1) / warpSize;
    float part_sum = sdata[tid];
    
    for(int offset=warpSize/2; offset > 0; offset >>= 1){
      // 把offset跟sum之间的xor进行交换
      // int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width=32);
      // 通过tid和laneMask之间进行xor操作，得到取值的tid,也就是这个tid对应的sum
      // 比如说，offset = 16, 二进制10000,进程1得到的就是10001,就是17,同理10001 xor 10000 = 1
      // 因此1和17交换了数据并相加了。这样做是的warp内的数据最终相加到0.
      // offset都取2的倍数。这样异或操作对小数来说是加，大数来说是减
      part_sum += __shfl_xor_sync(0xFFFFFFFF, part_sum, offset);
    }
    
    int warp_id = tid/warpSize;
    int lane = tid%warpSize;
    if(lane==0){
      //一共有NUM_WARPS个数据，这个值是小于32的
      sdata[warp_id] = part_sum;
    }
    __syncthreads();
    part_sum = (lane < NUM_WARPS)?sdata[tid]:0.0f;
    for(int offset=warpSize/2; offset > 0; offset >>= 1){
      part_sum += __shfl_xor_sync(0xFFFFFFFF, part_sum, offset);
    }
    __syncthreads();
    if(tid == 0){
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);

  CHECK_EQ(M, input.get_dim(0));
  if (config && config->stream) {
    my_matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    my_matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                              const_cast<float*>(output.ptr<float>()), M, K);
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}
}  // namespace kernel