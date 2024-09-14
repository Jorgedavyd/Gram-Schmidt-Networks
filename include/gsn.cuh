#ifndef GSN_CUH
#define GSN_CUH

#include <torch/extensions.h>

torch::Tensor forward(
    torch::Tensor& input,
    torch::Tensor&... weights,
) {};

torch::Tensor backward(
    torch::Tensor& input,
    torch::Tensor&... weights,
) {};


#endif //GSN_CUH
