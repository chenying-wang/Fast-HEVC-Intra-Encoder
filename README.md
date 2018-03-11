# Fast HEVC Intra Encoder

Fast HEVC Intra Encoder is an improvement on [HEVC Test Model](https://hevc.hhi.fraunhofer.de/svn/svn_HEVCSoftware) aimed at both CTU/CU partition and prediction mode decision in intra mode. In contrast to the current brute-force RDO search method, a faster approach of CTU/CU partition decision based on deep convolutional neural network and texture complexity will be implemented in [TensorFlow](https://www.tensorflow.org/).

# Build

## Linux

```
git clone git://github.com/chenying-wang/Fast-HEVC-Intra-Encoder.git
make -C src/HM-16.18/build/linux all
```

## Windows

```
git clone git://github.com/chenying-wang/Fast-HEVC-Intra-Encoder.git
MSBuild /t:Build src\HM-16.18\build\HM_vc2015.sln
```

# To-Do List

- [x] Extract CTU/CU modes of training data from HM encoder.
- [ ] Early-termination and early-split algorithm.
- [ ] Implement and train CNN in TensorFlow.
- [ ] Merge CNN into HM encoder and test.

# License

Copyright (c) 2018 Chenying Wang \<wangchenying@hust.edu.cn\>

Licensed under [MIT License](LICENSE)

---

HEVC Test Model

https://hevc.hhi.fraunhofer.de/svn/svn_HEVCSoftware

Copyright (c) 2010-2017, ITU/ISO/IEC

Licensed under the [BSD License](src/HM-16.18/COPYING)

---
