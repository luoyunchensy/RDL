# Relation Discrimination Learning for Zero-Shot Relation Triplet Extraction

This repository contains the implementation of RDL (Relation Discrimination Learning), a framework for zero-shot relation triplet extraction.

## Abstract
Zero-shot relation triplet extraction (ZeroRTE) extracts structured triplets from text without training on target relations. While large language models show promise for zero-shot tasks, existing approaches fail to effectively address the semantic drift of LLMs, leading to difficulties in distinguishing semantically similar relations in zero-shot scenarios. We propose Relation Discrimination Learning (RDL), which transforms relation extraction into discrimination by training models to select correct relations from candidate sets rather than performing direct extraction. This addresses the core challenge: distinguishing semantically similar relations. Extensive experiments show RDL achieves significant improvements over state-of-the-art baselines across different model scales and datasets, maintaining excellent performance in low-resource scenarios while requiring  no architectural modifications or multi-stage pipeline design. Our work demonstrates that relation discrimination capability is fundamental to successful zero-shot relation triplet extraction.

## Introduction
<img width="3272" height="1750" alt="main" src="https://github.com/user-attachments/assets/83174c2d-59fb-4d9d-baec-781e7a959316" />

## Results
<img width="1226" height="766" alt="ec3b9fc4cb857c49941700427818938e094562c6d2aa3bb742ce7c293716a37b" src="https://github.com/user-attachments/assets/e9d3011f-98ef-432d-b80e-c304246ee06e" />

<img width="1226" height="766" alt="iShot_2025-08-10_10 24 35" src="https://github.com/user-attachments/assets/da7b6738-1fab-43bb-8c89-6c66e5f0856c" />


