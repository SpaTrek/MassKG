

# MassKG v1.0 <img src='images/icon.png' align="right" height="200" /></a>

#  **Knowledge-based _In Silico_ Fragmentation and Annotation of Mass Spectra for Natural Products with MassKG**

### Bingjie Zhu<sup></sup>,  Zhenhao Li<sup></sup>, Zehua Jin<sup></sup>, ..., Jie Liao*,Xiaohui Fan*



MassKG is a knowledge-based computational tool for annotating mass spectra of natural products and discovering novel structures from crude extracts.

![Image text](images/overview.jpg)

## Requirements and Installation
This toolkit is written in Python programming languages. 
### Installation of MassKG (Python package)

[![faerun 0.4.7](https://img.shields.io/badge/faerun-0.4.7-blue)](https://pypi.org/project/faerun/0.4.7/) [![matplotlib 3.5.3](https://img.shields.io/badge/matplotlib-3.5.3-green)](https://github.com/matplotlib/matplotlib/) [![multiprocess 0.70.15](https://img.shields.io/badge/multiprocess-0.70.15-yellowgreen)](https://pypi.org/project/multiprocess/0.70.15/) [![numpy 1.21.3](https://img.shields.io/badge/numpy-1.21.3-yellow)](https://github.com/numpy/numpy/) [![openpyxl 3.1.2](https://img.shields.io/badge/openpyxl-3.1.2-orange)](https://pypi.org/project/openpyxl/3.1.2/) [![pandas 1.3.5](https://img.shields.io/badge/pandas-1.3.5-ff69b4)](https://github.com/pandas-dev/pandas/) [![scipy 1.7.3](https://img.shields.io/badge/scipy-1.7.3-purple)](https://github.com/scipy/scipy/) [![seaborn 0.12.2](https://img.shields.io/badge/seaborn-0.12.2-9cf)](https://github.com/mwaskom/seaborn/) [![tmap 1.0.6](https://img.shields.io/badge/tmap-1.0.6-inactive)](https://pypi.org/project/tmap/1.0.6/) [![torch 1.13.1+cu117](https://img.shields.io/badge/torch-1.13.1%2Bcu117-blueviolet)](https://pytorch.org/) [![rdkit 2023.3.2](https://img.shields.io/badge/rdkit-2023.3.2-lightgrey)](https://www.rdkit.org/) [![scikit-learn 1.0.2](https://img.shields.io/badge/scikit--learn-1.0.2-brown)](https://github.com/scikit-learn/scikit-learn/)





# Introduction to MassKG



Liquid chromatography coupled with tandem mass spectrometry (LC-MS) is a powerful analytical technique used for the identification of metabolites from complicated sources. The general procedure for annotating mass spectrometry data is based on comparing experimental data with mass spectra of standard substance. However, currently, there are few reference LC-MS libraries for natural products due to their structural diversity, and publicly available mass spectrometry databases exhibit biases in terms of scale, coverage, and quality. Here, we propose MassKG, an algorithm that combines a knowledge-based fragmentation strategy and a deep learning-based molecule generation model to expand the reference spectra library of NPs with the more abundant chemical space. Specifically, MassKG has collected 407,720 known NP structures and, based on this, generated 266,353 new structures with chemical language models for the discovery of potential novel new compounds. Moreover, MassKG shows outstanding performance of spectra annotation compared to state-of-the-art algorithms, such as MSFINDER, CFM-ID, and MetFrag. To facilitate the usage, MassKG has been implemented as a web server for MS data annotation with a user-friendly interface, an automatic report, and the fragment tree visualization. Finally, the interpretation ability of MassKG is comprehensively validated through composition analysis and MS annotation of _Ginkgo Biloba_, _Codonopsis pilosula_, and _Astragalus membranaceus_.

* MassKG has been implemented at https://xomics.com.cn/masskg
* The chemical lauguage model please refer to https://github.com/skinnider/low-data-generative-models
* Complete TMAP of MassKG NP space ([Google Drive](https://drive.google.com/drive/folders/1U_ne24Be8vxwLpBZ_BHa-idXMowl0KMj))
## About
Should you have any questions, please feel free to contact the author of the manuscript, Mr. Bingjie Zhu (zhubj@zju.edu.cn).

## References

