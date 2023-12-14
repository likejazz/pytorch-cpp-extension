#ifndef __CUDAEXPR_H
#define __CUDAEXPR_H

#include <torch/torch.h>

torch::Tensor add_one(const torch::Tensor &input);

#endif // __CUDAEXPR_H
