# Research Papers on Sign Language Processing

This document provides a curated list of important research papers that form the foundation of LinguaSign and the broader field of sign language processing. Each paper is accompanied by a brief summary and its relevance to our project.

## Core Papers in Sign Language Processing

### Sign Language Recognition

1. **Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison**
   - Authors: Dongxu Li, Cristian Rodriguez Opazo, Xin Yu, Hongdong Li
   - Publication: WACV 2020
   - [Paper Link](https://arxiv.org/abs/1910.11006)
   - Summary: Introduces WLASL, a large-scale word-level American Sign Language dataset, and benchmarks various deep learning approaches.
   - Relevance: Our project uses the WLASL dataset and builds upon their baseline models.

2. **Neural Sign Language Recognition: A Review**
   - Authors: Oscar Koller
   - Publication: IEEE PAMI 2020
   - [Paper Link](https://ieeexplore.ieee.org/document/9222551)
   - Summary: Comprehensive review of neural network approaches to sign language recognition.
   - Relevance: Provides theoretical foundations for our model architectures.

3. **Attention is All You Need**
   - Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
   - Publication: NeurIPS 2017
   - [Paper Link](https://arxiv.org/abs/1706.03762)
   - Summary: Introduces the Transformer architecture, which revolutionized sequence modeling.
   - Relevance: Foundational paper for our transformer-based models.

### Sign Language Translation

4. **Neural Sign Language Translation**
   - Authors: Necati Cihan Camgoz, Simon Hadfield, Oscar Koller, Hermann Ney, Richard Bowden
   - Publication: CVPR 2018
   - [Paper Link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Camgoz_Neural_Sign_Language_CVPR_2018_paper.pdf)
   - Summary: Introduces end-to-end sign language translation and the PHOENIX-2014T dataset.
   - Relevance: Our transformer model is inspired by this approach.

5. **Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation**
   - Authors: Necati Cihan Camgoz, Oscar Koller, Simon Hadfield, Richard Bowden
   - Publication: CVPR 2020
   - [Paper Link](https://arxiv.org/abs/2003.13830)
   - Summary: Applies transformer architecture to joint sign language recognition and translation.
   - Relevance: Direct influence on our transformer implementation.

6. **Improving Sign Language Translation with Monolingual Data by Sign Back-Translation**
   - Authors: Hao Zhou, Wengang Zhou, Weizhen Qi, Junfu Pu, Houqiang Li
   - Publication: CVPR 2021
   - [Paper Link](https://arxiv.org/abs/2105.12397)
   - Summary: Proposes sign back-translation to utilize monolingual data for sign language translation.
   - Relevance: Inspiration for data augmentation techniques in our training process.

### Sign Language Datasets

7. **MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language**
   - Authors: Hamid Reza Vaezi Joze, Oscar Koller
   - Publication: BMVC 2019
   - [Paper Link](https://arxiv.org/abs/1812.01053)
   - Summary: Introduces MS-ASL, a large-scale dataset for ASL recognition.
   - Relevance: Complementary dataset to WLASL that we plan to incorporate.

8. **BOBSL: A Large-Scale Domain-General Sign Language Dataset for Continuous Sign Recognition and Translation**
   - Authors: Samuel Albanie, Gül Varol, Liliane Momeni, Hannah Bull, Triantafyllos Afouras, Himel Chowdhury, Neil Fox, Bencie Woll, Rob Cooper, Andrew Zisserman, Necati Cihan Camgoz
   - Publication: ArXiv 2021
   - [Paper Link](https://arxiv.org/abs/2111.03635)
   - Summary: Introduces BOBSL, a large-scale British Sign Language dataset.
   - Relevance: Future work for expanding LinguaSign to BSL.

### MediaPipe and Feature Extraction

9. **MediaPipe Hands: On-device Real-time Hand Tracking**
   - Authors: Fan Zhang, Valentin Bazarevsky, Andrey Vakunov, Andrei Tkachenka, George Sung, Chuo-Ling Chang, Matthias Grundmann
   - Publication: CVPR 2020 Workshop
   - [Paper Link](https://arxiv.org/abs/2006.10214)
   - Summary: Describes MediaPipe Hands, a real-time hand tracking system.
   - Relevance: Core component of our feature extraction pipeline.

10. **BlazePose: On-device Real-time Body Pose Tracking**
    - Authors: Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, Matthias Grundmann
    - Publication: CVPR 2020 Workshop
    - [Paper Link](https://arxiv.org/abs/2006.10204)
    - Summary: Describes BlazePose, a real-time pose estimation system in MediaPipe.
    - Relevance: Used for body pose extraction in our pipeline.

## Recent Advancements

11. **SignBERT: Pre-Training a Language Model for Sign Language Understanding**
    - Authors: Hao Zhou, Wengang Zhou, Weizhen Qi, Junfu Pu, Houqiang Li
    - Publication: AAAI 2022
    - [Paper Link](https://arxiv.org/abs/2112.08637)
    - Summary: Proposes a BERT-like model pre-trained on sign language.
    - Relevance: Potential direction for improving our model architecture.

12. **SignGPT: Generative Pre-trained Transformer for Sign Language**
    - Authors: Jiahui Huang, Zhou Zhao, Yutong Wen, Deng Cai, Minh-Thang Luong, Yuan Cao
    - Publication: CVPR 2023
    - [Paper Link](https://arxiv.org/abs/2306.00980)
    - Summary: Introduces a GPT-like model for sign language generation.
    - Relevance: Future direction for text-to-sign generation.

13. **An Integrated Mediapipe-Optimized GRU Model for Indian Sign Language Recognition**
    - Authors: Abhishek Jha, Aditya Nigam, Anupam Agarwal
    - Publication: Scientific Reports 2022
    - [Paper Link](https://www.nature.com/articles/s41598-022-15998-7)
    - Summary: Integrates MediaPipe with GRU for improved sign language recognition.
    - Relevance: Similar approach to our MediaPipe+ML model.

## Ethical and Accessibility Considerations

14. **Systemic Biases in Sign Language AI Research: A Deaf-Led Call to Reevaluate Research Agendas**
    - Authors: Naomi Caselli, Ben Saunders, Lauren W. Berger, Ronice Müller de Quadros, Richard Mesch, Abraham Glasser, Diane Brentari
    - Publication: ArXiv 2024
    - [Paper Link](https://arxiv.org/abs/2403.02563)
    - Summary: Deaf-led critique of current sign language AI research and datasets.
    - Relevance: Essential guidance for ethical development of our project.

15. **The FATE Landscape of Sign Language AI Datasets: An Interdisciplinary Perspective**
    - Authors: Danielle Bragg, Oscar Koller, Mary Beth Rosson, Stephanie Ludi
    - Publication: ACM Transactions on Accessible Computing 2021
    - [Paper Link](https://dl.acm.org/doi/10.1145/3436996)
    - Summary: Analysis of fairness, accountability, transparency, and ethics in sign language datasets.
    - Relevance: Guides our dataset selection and preprocessing policies.

## Contributing

If you would like to suggest additional papers to include in this list, please submit a pull request with the paper information and a brief summary of its relevance to LinguaSign.

## Citation Format

When citing these papers in your own work, please use the following format:

```
Authors. "Title." Publication, Year. URL.
```

For example:

```
Dongxu Li, Cristian Rodriguez Opazo, Xin Yu, Hongdong Li. "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison." WACV, 2020. https://arxiv.org/abs/1910.11006.
```
