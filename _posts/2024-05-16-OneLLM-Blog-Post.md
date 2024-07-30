---
title: 'A comprehensive introduction to OneLLM: One Framework to Align All Modalities with Language'
date: 2024-05-16
permalink: /onellm/
tags:
  - OneLLM
  - language model
---

# Motivation

Deep Learning has revolutionized various fields like image recognition and language processing by handling complex data structures and finding patterns within those.
However, as tasks become increasingly multimodal, integrating different data sources like text, images, audio, and video poses new challenges. Traditional models, typically focusing on fewer than three modalities, fall short of this transformation.
OneLLM aims to create a unified framework to handle and align multiple modalities with language, addressing inefficiencies in existing models that rely on limited modality-specific encoders. This blogpost explores OneLLM's development, introducing a framework that aligns eight modalities with language using a unified multimodal encoder and a progressive alignment pipeline.
As we will later showcase, this approach shows significant performance improvements in captioning and reasoning, advancing the field of multimodality. Join us to explore OneLLM's potential!


# Table of Contents

1. Large Language Model Background
	- LLM
	- Vision LLM
	- MLLM
2. OneLLM
	- architecture
3. Experiments
	- Quantitative & Qualitative Analysis
4. Ablation
5. Pros & Cons 
6. Conclusion


LLM
------
Large Language Models (LLM) are artificial neural networks that use the transformer architecture. Which was introduced in 2017 by Vaswani et al. in the paper “Attention is all you need” [1]. Where they used it for translation purposes. 
Nowadays LLM excel at general natural language processing tasks like question answering, reasoning or prompt generating. The reason behind is the extraordinary parallelizing and scaling capabilities of transformers which allow them to be trained on large amounts of data in a relatively short time.


MLLM
------
However, the modalities of these transformers and their ability to capture various inputs is limited. Along comes the domain of MLLMs to apply a solution to the abundance of modalities. Three key parts of the MLLMs architecture are responsible for managing consistent inputs: encoders and projection modules for each modality, and the Multimodal Large Language Model in order of application. 
While this is largely the same structure as a regular LLM, it differs in scalability, although only in relative comparison.
Examples of MLLMs can be seen in recent works such as ChatBridge, which takes advantage of Perceiver, a transformer revolutionizing the AI domain in 2017, able to work with multiple common modalities and superior efficiency due to self- and cross-attention mechanisms. 

OneLLM
======
While MLLMs prove to be the superior choice when trying to work with multiple modalities, its scalability is rather weak, with models rarely exceeding three modalities at once. The key point is how the input into the LLM is being manipulated to align the different modalities, yet still produce uniform inputs to produce relevant outputs. 
OneLLM balances these issues out with key components that have been unprecedented in the domain of Deep Learning previously.

Tokenizers
------
While traditionally each modality gains access to its very own encoder, OneLLM instead opts for modality-specific tokenizers to effortlessly convert input into sequences. Depending on the modalities, 1D and 2D convolution layers are being used to generate fix-length tokens out of said sequences of signals. To notate the modality tokens, it is defined as $x \in \mathds{R}^{L \times D}$ where L is the sequence length and D defines the token dimension. 

The universal encoder
------
The encoder differs quite strongly from traditional models. To leverage the strong cross-modality capabilities that vision models have, CLIP-Vit is used in encoding the tokenized input data. Note that a model is used here as an encoder, with frozen parameters following previous works [51] Lu et al. and [103] Zhang et al.

The Universal Projection Module (UPM)
------
is one of the key components to unifying underlying differences of modalities. To realize this, the Projection Module includes experts denoted as ${P_K}$ where K denotes the number of experts, that are pretrained on image-text data, greatly enhancing the scalability when employing multiple modality-LLM experts. Also included within the UMP is a modality router R  that manages each experts’ contributions.
The router is based on a neural network, computing how the tokens are to be distributed among the experts.\\\textbf{[Need details on Router]}


Experiments
------
To quantitatively compare OneLLM to other multimodal models currently available, specific training instructions have been set. These include three projection experts with each eight Transformers and 88M parameters, as well as modality tokens with a size of $\mathds{R}^{30 \times 1024}$. As LLM, LLaMA2 with 7 billion parameters was chosen. The results have been split for each modality, comparing each performance with generalist MLLMs and modality-specific LLMs available. 

Especially in tasks such as Image-Text and Point-Cloud-Text Evaluations or IMU- and fMRI-Text Evaluations, OneLLM shows clear superiority, indicating its abilities to adapt to diverse modalities.

Despite not being specifically trained on datasets as image captioning or audio captioning, OneLLM yet consistently outperforms existing models. Its zero-shot capabilites, to correctly perform tasks without having seen similar examples in training before, is quite astonishing and does surpass fully fine-tuned modality-specific models. Specifically in audio-text evaluation, it shows on-par capabilties with Pengi when evaluated on Clotho AQA, a crowd-sourced Audio Question Answering dataset.

Ablation
------
Even though OneLLM showed superiority within the evaluation Experiments, some key design features are subject to change.

An important feature discussed is seperate training in comparison to joint training, where models are restricted or allowed to pass learned knowledge between each other, respectively.

While an evaluation image training shows comparable results, there is a clear drop of quality in results when it comes to seperately trained models for audio and video tasks. Specifically modalities which lack diversity in data sets present issues to be averted when applying joint training, allowing to transfer knowledge across modalities.
% verschiedene design parts ansprechen, \subsubsections einrichten
Solidifying the earlier hypothesis that image-text modality is a robust option to align any other modalities with it, performance measures indicate similar results. If instead of using image-text-pretraining OneLLM would be trained on another random modality, performance of image and video specifically will drop significantly by up to 15\%. This again reinforces the idea of image-text being a strong multimodal alignment factor, caused by its abundance of available data.

Number Projection Experts
------

Router Type
------

Qualitative Analysis
------

Conclusion
------