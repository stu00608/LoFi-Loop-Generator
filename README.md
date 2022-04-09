## Environment

* GTX 1080Ti with CUDA 11.0 and cuDnn 8.0.5

```
nvcc -V
```

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Thu_Jun_11_22:26:38_PDT_2020
Cuda compilation tools, release 11.0, V11.0.194
Build cuda_11.0_bu.TC445_37.28540450_0
```

```
cat /usr/local/cuda-11.0/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

```
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 0
#define CUDNN_PATCHLEVEL 5
--
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#endif /* CUDNN_VERSION_H */
```

* Conda with Python 3.8.12 and the packages listed in `requirements.txt`

## IDEA

Some description :

    Data :                js-fakes https://github.com/omarperacha/js-fakes
                          It's a midi dataset included 500 4 tracks midi files.

    Representation :      Read file as pianoroll, every beat will represent as an array with shape (resolution, 128).
                          Then make the pianoroll matrix to a time series data. The encoded beat data is our "word". Use
                          Tokenizer to map most frequently used 200 data into a dictionary, and map it as a integer. Finally
                          turn the integer to one_hot array.
  ```
  pitch          
    4| 0 0 0 0 0 0
    3| 1 1 0 0 0 0                       Tokenizer
    2| 0 0 1 1 0 0 ... ---> 3-2#2-2#1-2 -----------> int -> one_hot array.
    1| 0 0 0 0 1 1
    0| 0 0 0 0 0 0
                beat
  ```
