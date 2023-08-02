# Minimal Solvers for 3D Scan Alignment With Pairs of Intersecting Lines
Andre Mateus<sup>1</sup>, Srikumar Ramalingam<sup>2</sup>, and Pedro Miraldo<sup>1</sup>

<sup>1</sup>Instituto Superior Tecnico, Lisboa <sup>2</sup>Google Research, NY 
<br />
E-Mail: {andre.mateus,pedro.miraldo}@tecnico.ulisboa.pt


<img src="imgs/fig0.gif" width="200" />

This project provides minimal solvers for 3D registration using intersecting lines.

If you want to use solvers.cpp/hpp file, please cite:
```
@InProceedings{Mateus_2020_CVPR,
author = {Mateus, Andre and Ramalingam, Srikumar and Miraldo, Pedro},
title = {Minimal Solvers for 3D Scan Alignment With Pairs of Intersecting Lines},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
and
```
@article{Mateus_2023_IJCV,
  title={Fast and Accurate 3D Registration from Line Intersection Constraints},
  author={Mateus, Andr{\'e} and Ranade, Siddhant and Ramalingam, Srikumar and Miraldo, Pedro},
  journal={International Journal of Computer Vision (IJCV)},
  pages={1--26},
  year={2023}
}
```

For solverPoints.cpp/hpp file, please cite:
```
@InProceedings{Miraldo_2019_CVPR,
author = {Miraldo, Pedro and Saha, Surojit and Ramalingam, Srikumar},
title = {Minimal Solvers for Mini-Loop Closures in 3D Multi-Scan Alignment},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

For solverLines.cpp/hpp file, please cite:
```
@InProceedings{henrikstewenius2005solutions,
  title={Solutions to minimal generalized relative pose problems},
  author={HenrikStew{\'e}nius, MagnusOskarsson and Astr{\"o}m, Kalle and Nist{\'e}r, David},
  year={2005},
  booktitle={OMNIVIS}
}
```

### Build examples

We provide examples on how to use the solvers.
The only dependency is Eigen3, to install it run 
```
sudo apt-get install libeigen3-dev
```
Then, compile the code
```
cd 3DMinRegLineIntersect
mkdir build && cd build
cmake ..
make -j$(nthreads)
```
