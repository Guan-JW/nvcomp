#pragma once
#include "util.h"

class BatchData
{
public:
    BatchData(
        char* device_data,
        const size_t in_bytes,
        const size_t batch_size,
        const size_t chunk_size,
        cudaStream_t stream) :
        m_ptrs(),
        m_sizes(),
        m_data(device_data),
        m_size(batch_size)
    {
        // Setup an array of pointers to the start of each chunk
        void** host_uncompressed_ptrs;
        cudaMallocHost((void**)&host_uncompressed_ptrs, sizeof(size_t) * batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            host_uncompressed_ptrs[i] = m_data + chunk_size * i;
        }

        cudaMalloc((void**)&m_ptrs, sizeof(size_t) * batch_size);
        cudaMemcpyAsync(m_ptrs, host_uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

        size_t* host_uncompressed_bytes;
        cudaMallocHost((void**)&host_uncompressed_bytes, sizeof(size_t) * batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            if (i + 1 < batch_size) {
                host_uncompressed_bytes[i] = chunk_size;
            } else {
                // last chunk may be smaller
                host_uncompressed_bytes[i] = in_bytes - (chunk_size*i);
            }
        }
        
        cudaMalloc((void**)&m_sizes, sizeof(size_t) * batch_size);
        cudaMemcpyAsync(m_sizes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

        // Free host memory
        cudaStreamSynchronize(stream);
        cudaFreeHost(host_uncompressed_ptrs);
        cudaFreeHost(host_uncompressed_bytes);
    }

    BatchData(const size_t max_out_bytes, const size_t batch_size, cudaStream_t stream) :
        m_ptrs(),
        m_sizes(),
        m_data(),
        m_size(batch_size)
    {

        // allocate output space on the device
        void ** host_compressed_ptrs;
        cudaMallocHost((void**)&host_compressed_ptrs, sizeof(size_t) * batch_size);
        for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
            CUDA_CHECK(cudaMalloc(&host_compressed_ptrs[ix_chunk], max_out_bytes));
        }

        CUDA_CHECK(cudaMalloc((void**)&m_ptrs, sizeof(size_t) * batch_size));
        cudaMemcpyAsync(
            m_ptrs, host_compressed_ptrs, 
            sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

        // allocate space for compressed chunk sizes to be written to
        CUDA_CHECK(cudaMalloc((void**)&m_sizes, sizeof(size_t) * batch_size));
        CUDA_CHECK(cudaMalloc((void**)&m_data, max_out_bytes * batch_size));

        // Free host memory
        cudaStreamSynchronize(stream);
        cudaFreeHost(host_compressed_ptrs);
    }

    char* data() 
    {
        return m_data;
    }

    void** ptrs()
    {
        return m_ptrs;
    }

    size_t* sizes()
    {
        return m_sizes;
    }

    size_t size() const
    {
        return m_size;
    }

private:
    void ** m_ptrs;
    size_t * m_sizes;
    char * m_data;
    size_t m_size;  // batch size
};
