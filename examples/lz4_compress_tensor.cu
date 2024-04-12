#include "BatchDataTensor.h"
#include "util.h"
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include "lz4.h"
#include "lz4hc.h"
#include "nvcomp/lz4.h"
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


BatchData execute_lz4compress(char* device_input_data, 
      const size_t in_bytes, const size_t batch_size, const size_t chunk_size) {
  // Start compression logic 
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // build up metadata
  BatchData input_data(device_input_data, in_bytes, batch_size, chunk_size, stream);
  std::cout << "in_bytes: " << in_bytes << "; batch_size: " << batch_size << "; chunk_size: " << chunk_size << std::endl;
  
  // record time
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  nvcompStatus_t status = nvcompBatchedLZ4CompressGetTempSize(
      input_data.size(),
      chunk_size,
      nvcompBatchedLZ4DefaultOpts,
      &comp_temp_bytes);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedLZ4CompressGetTempSize() not successful");
  }
  std::cout << "comp_temp_bytes: " << comp_temp_bytes << std::endl;

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_out_bytes;
  status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedLZ4CompressGetMaxOutputChunkSize() not successful");
  }
  std::cout << "max_out_bytes: " << max_out_bytes << std::endl;

  BatchData compress_data(max_out_bytes, input_data.size(), stream);
  
  std::cout << input_data.size() << std::endl;
  std::cout << input_data.ptrs() << "; " << input_data.sizes() << std::endl;
  std::cout << compress_data.ptrs() << std::endl;
  std::cout << "; " << compress_data.sizes() << std::endl;
  // std::cout << "; " << compress_data.data() << std::endl;
  status = nvcompBatchedLZ4CompressAsync(
      input_data.ptrs(),
      input_data.sizes(),
      chunk_size,
      input_data.size(),
      d_comp_temp,
      comp_temp_bytes,
      compress_data.ptrs(),
      compress_data.sizes(),
      nvcompBatchedLZ4DefaultOpts,
      stream);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedLZ4CompressAsync() failed.");
  }
  

  cudaEventRecord(end, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // free compression memory
  cudaFree(d_comp_temp);

  float ms;
  cudaEventElapsedTime(&ms, start, end);
  
  // compute compression ratio
  size_t * host_compressed_bytes;
  cudaMallocHost((void**)&host_compressed_bytes, sizeof(size_t) * batch_size);

  cudaMemcpy(
    host_compressed_bytes,
    compress_data.sizes(),
    sizeof(size_t) * batch_size,
    cudaMemcpyDeviceToHost);
  size_t comp_bytes = 0;
  for (size_t i = 0; i < batch_size; i ++) {
    comp_bytes += host_compressed_bytes[i];
  }

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)in_bytes / comp_bytes << std::endl;
  std::cout << "compression time (ms): " << ms << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaStreamDestroy(stream);

  return compress_data;
}

BatchData lz4compress_wrapper(torch::Tensor input_tensor) {
  // Ensure the input tensor is on the correct device, has the correct data type, and is contiguous
  // input_tensor = input_tensor.contiguous().to(torch::kCPU, torch::kUInt8);
  CHECK_INPUT(input_tensor);
  std::cout << "Total number of elements in input_tensor: " << input_tensor.numel() << std::endl;

  // Cast the tensor data (with INT) to char* for byte-wise compression
  char* device_input_data = reinterpret_cast<char*>(input_tensor.data_ptr<int>());

  // Calculate the total byte size of the tensor's data
  size_t in_bytes = input_tensor.numel() * sizeof(int);

  const size_t chunk_size = 1 << 16;
  const size_t batch_size = (in_bytes + chunk_size - 1) / chunk_size;

  return execute_lz4compress(device_input_data, in_bytes, batch_size, chunk_size);
}

int main() {
    // Set device to CUDA
    torch::Device device(torch::kCUDA);

    // Create a tensor of type int32 with random values
    // Specify the desired size of the tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto input_tensor = torch::randint(std::numeric_limits<int>::max(), {1024}, options);

    // Call your lz4compress_wrapper function with the created tensor
    auto compress_data = lz4compress_wrapper(input_tensor);
    
    return 0;
}