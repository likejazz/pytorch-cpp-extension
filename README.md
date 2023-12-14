Demonstration of using PyTorch with cmake
=========================================

To compile:

```bash
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -GNinja ..
```

To run:

```bash
./test.py
```

## References
- <https://pytorch.org/cppdocs/installing.html>
- <https://github.com/ita9naiwa/attention-impl>
