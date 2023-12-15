#include <torch/torch.h>
#include <iostream>

#include "cudaexpr.h"

int main() {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! running on GPU." << std::endl;
        device = torch::kCUDA;
    }

    torch::Tensor a = torch::rand({2, 3}).to(device);
    torch::Tensor b = add_one(a);
    std::cout << a << std::endl;
    std::cout << b << std::endl;

    return 0;
}