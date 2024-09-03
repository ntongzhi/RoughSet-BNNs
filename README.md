This article was published in the journal Applied Soft Computing under the title "Learning trustworthy model from noisy labels based on rough set for surface defect detection". 
Please cite our article:
@article{NIU2024112138,
title = {Learning trustworthy model from noisy labels based on rough set for surface defect detection},
journal = {Applied Soft Computing},
volume = {165},
pages = {112138},
year = {2024},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2024.112138},
url = {https://www.sciencedirect.com/science/article/pii/S1568494624009128},
author = {Tongzhi Niu and Zhenrong Wang and Weifeng Li and Kai Li and Yuwei Li and Guiyin Xu and Bin Li},
keywords = {Trustworthy model, Surface defect detection, Noisy label, Rough set, Bayesian neural networks},
abstract = {In surface defect detection, some regions remain ambiguous and cannot be distinctly classified as abnormal or normal. This challenge is exacerbated by subjective factors, including workers’ emotional fluctuations and judgment variability, resulting in noisy labels that lead to false positives and missed detections. Current methods depend on additional labels, such as clean and multi-labels, which are both time-consuming and labor-intensive. To address this, we utilize Rough Set theory and Bayesian neural networks to learn a trustworthy model from noisy labels for Surface Defect Detection. Our approach features a novel pixel-level representation of suspicious areas using lower and upper approximations, and a novel loss function that emphasizes both precision and recall. The Pluggable Spatially Bayesian Module (PSBM) we developed enhances probabilistic segmentation, effectively capturing uncertainty without requiring extra labels or architectural modifications. Additionally, we have devised a ‘defect discrimination confidence’ metric to better quantify uncertainty and assist in product quality grading. Without the need for extra labeling, our method significantly outperforms state-of-the-art techniques across three types of datasets and enhances seven types of classic networks as a pluggable module, without compromising real-time computing performance. For further details and implementation, our code is accessible at https://github.com/ntongzhi/RoughSet-BNNs.}
}
