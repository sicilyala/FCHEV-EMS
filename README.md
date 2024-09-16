#  FCHEV-EMS: Health-considered energy management strategy for fuel cell hybrid electric vehicle based on improved soft actor critic algorithm adopted with Beta policy.
## Overview

The original implementation of **Health-considered energy management strategy for fuel cell hybrid electric vehicle based on improved soft actor critic algorithm adopted with Beta policy.**


## Abstract

Deep reinforcement learning-based energy management strategy (EMS) is essential for fuel cell hybrid electric vehicles to reduce hydrogen consumption, improve health performance and maintain charge. This is a complex nonlinear constrained optimization problem. In order to solve the problem of high bias caused by the inconsistency between the infinite support of stochastic policy and the bounded physics constraints of application scenarios, this paper proposes the Beta policy to improve standard soft actor critic (SAC) algorithm. This work takes hydrogen consumption, health degradation of both fuel cell system and power battery, and charge margin into consideration to design an EMS based on the improved SAC algorithm. Specifically, an appropriate tradeoff between money cost during driving and charge margin is firstly determined. Then, optimization performance differences between the Beta policy and the standard Gaussian policy are presented. Thirdly, ablation experiments of health constraints are conducted to show the validity of health management. Finally, comparison experiments indicate that, the proposed strategy has a 5.12% performance gap with dynamic programming-based EMS with respect to money cost, but is 4.72% better regarding to equivalent hydrogen consumption. Moreover, similar performances in validation cycle demonstrate good adaptability of the proposed EMS.


## Data

1. **Driving Cycles can be found [here](https://github.com/sicilyala/project-data/tree/main/standard_driving_cycles).**

2. **Power system data can be found [here](https://github.com/sicilyala/project-data/tree/main/FCHEV_data).**


## Citation
**BibTex**
```
@article{chen2023health,
  title={Health-considered energy management strategy for fuel cell hybrid electric vehicle based on improved soft actor critic algorithm adopted with Beta policy},
  author={Chen, Weiqi and Peng, Jiankun and Chen, Jun and Zhou, Jiaxuan and Wei, Zhongbao and Ma, Chunye},
  journal={Energy Conversion and Management},
  volume={292},
  pages={117362},
  year={2023},
  publisher={Elsevier}
}
```
