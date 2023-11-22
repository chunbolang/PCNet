# Progressive Parsing and Commonality Distillation for Few-Shot Remote Sensing Segmentation

This repository contains the source code for our paper "*Progressive Parsing and Commonality Distillation for Few-Shot Remote Sensing Segmentation*" by Chunbo Lang, Junyi Wang, Gong Cheng, Binfei Tu, and Junwei Han.

> **Abstract:** *In recent years, few-shot segmentation (FSS) has received widespread attention from scholars by virtue of its superiority in low-data regimes. Most existing research focuses on natural image processing, and very few studies are dedicated to the practical but challenging topic of remote sensing image understanding. Related experimental results show that directly transferring the previously proposed framework to the current domain is prone to produce unsatisfactory results with incomplete objects and irrelevant distractors. Such phenomena can be attributed to the lack of modules specifically designed for the complex characteristics of remote sensing images, e.g., great intra-class diversity and low target-background contrast. In this article, we propose a conceptually simple and easy-to-implement framework to tackle the aforementioned problems. Specifically, our innovative design embodies two main aspects: 1) the support mask is progressively parsed into multiple valuable subregions that can be further exploited to compute local descriptors with segmentation cues about intractable parts; and 2) the base-class memories stored in the meta-training phase are replayed and leveraged for the distillation of novel-class prototypes, where the commonalities between classes are adequately explored, more in line with the concept of learning to learn. These two components, i.e., the progressive parsing module and commonality distillation module, contribute to each other and together constitute the proposed PCNet. We conduct extensive experiments on the standard benchmark to evaluate segmentation performance in few-shot settings. Quantitative and qualitative results illustrate that our PCNet distinctly outperforms previous FSS approaches and sets a new state-of-the-art.*

## ▶️ Getting Started

Please refer to our [R2Net](https://github.com/chunbolang/R2Net) code repository for environment setup and result reproduction.

## 📖 BibTex

If you find this repository useful for your publications, please consider citing our paper.

```bibtex
@article{lang2023pcnet,
	title={Progressive Parsing and Commonality Distillation for Few-shot Remote Sensing Segmentation},
	author={Lang, Chunbo and Wang, Junyi and Cheng, Gong and Tu, Binfei and Han, Junwei},
	journal={IEEE Transactions on Geoscience and Remote Sensing},
	volume={61},
	pages={1-10},
	year={2023},
}
  
@article{lang2023r2net,
	title={Global Rectification and Decoupled Registration for Few-Shot Segmentation in Remote Sensing Imagery},
	author={Lang, Chunbo and Cheng, Gong and Tu, Binfei and Han, Junwei},
	journal={IEEE Transactions on Geoscience and Remote Sensing},
	volume={61},
	pages={1-11},
	year={2023},
}
```
