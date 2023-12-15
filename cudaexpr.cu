#include <c10/cuda/CUDAException.h>

#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/library.h>

template<class T>
__global__ void add_one_kernel(const T *const input, T *const output, const int64_t N) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        output[i] = input[i] + 1;
    }
}

torch::Tensor add_one(const torch::Tensor &input) {
    auto output = torch::zeros_like(input);

    AT_DISPATCH_ALL_TYPES(
            input.scalar_type(), "add_one_kernel", [&]() {
                const auto block_size = 128;
                const auto num_blocks = std::min(
                        65535L,
                        (input.numel() + block_size - 1) / block_size
                );
                add_one_kernel<<<num_blocks, block_size>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        input.numel()
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
    );

    return output;
}


// @formatter:off
// CMake Interface
TORCH_LIBRARY(cudaexpr, m) {
    m.def("add_one(Tensor input) -> Tensor");
    m.impl("add_one", c10::DispatchKey::CUDA, TORCH_FN(add_one));
}

// setup.py Interface
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    m.def("add_one", &add_one);
//}
// @formatter:off