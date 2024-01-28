# The normalised Distance from Unimodality (nDFU)

[Distance from unimodality (DFU)](https://github.com/ipavlopoulos/dfu/) has been found to correlate well with human judgment for the assessment of polarized opinions. However, its un-normalized nature makes it less intuitive and somewhat difficult to exploit in machine learning (e.g., as a supervised signal). 

This work introduces a normalized version of this measure (nDFU) that leads to better assessment of the degree of polarization. Part of this work is a methodology for K-class text classification, based on nDFU, that exploits polarized texts in the dataset. Such polarized instances are assigned to a separate K+1 class, so that a K+1-class classifier is trained. 

An empirical analysis on three datasets for toxic language detection, shows that nDFU can be used to model polarized annotations and prevent them from harming the classification performance. Finally, we further exploit nDFU to specify conditions that could explain polarization given a dimension and present text examples that polarized the annotators when the dimension was gender and race.

You may find the preprint of this [article in Oper Review](https://openreview.net/pdf?id=DKNaMP33ZL) and you can use the notebook in this repository to reproduce the experiments of the article (the datasets are not shared here, but can be retrieved).

Please cite this work as:
```
@inproceedings{pavlopoulos-likas-ndfu,
  title={Polarized Opinion Detection Improves the Detection of Toxic Language},
  author={Pavlopoulos, John and Likas, Aristidis},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics},
  year={2024}
}
```

Please consider citing the original [DFU article](https://github.com/ipavlopoulos/dfu) as:
```
@article{pavlopoulos-likas-2022,
    title = "Distance from Unimodality for the Assessment of Opinion Polarization",
    author = "Pavlopoulos, John  and Likas, Aristidis",
    journal = "Cognitive Computation",
    doi = "10.1007/s12559-022-10088-2",
    year = "2022",
}
```
