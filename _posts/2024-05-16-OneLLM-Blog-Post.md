---
title: 'OneLLM: One Framework to Align All Modalities with Language'
date: 2024-05-16
permalink: /posts/2012/08/blog-post-1/
tags:
  - OneLLM
---

Deep Learning has revolutionized a number of fields in the last years by providing powerful capabilities for tasks that involve complex data structures and patterns, such as image recognition, language processing, and more. 

However, as Deep Learning proves to be effective in further areas over time, tasks become increasingly multimodal. This multimodality introduces new issues that come in the form of integrating and processing different data sources, such as text, images, audio or video. Using traditional models, which – for the most part – do not focus on any more than three modalities, fall short to the tasks at hand.
Therefore, the motivation of OneLLM lies in creating a unified framework that allows to handle and align multiple modalities with language in order to address multimodal understanding. Doing so addresses the aforementioned inefficiencies that arise in existing models, heavily relying on modality-specific encoders which prove to be limited to a few common modalities. 

In this blogpost, we intend to explore the advancements presented during the development of OneLLM, introducing a unified framework that successfully aligns eight modalities to language. Overcoming the limitations of traditional models becomes possible through the use of a unified multimodal encoder and a progressive alignment pipeline, showing immense performance in captioning and reasoning. Join us as we explore the inner works and potential of OneLLM, advancing the field of multimodality.

Recap of (M)LLMs
======
So, what makes Multimodal Large Language Models (MLLM) so popular and how do they compare to Large Language Models (LLM)? To answer this, we start at a relatable example: an LLM, specifically designed for vision tasks.

LLM
------
At its core, a Large Language Model is a deep learning model that is pre-trained on immensely large data sets. It features three main parts: an Image Encoder, a Projection Module and the pre-trained Vision LLM.

The Encoder is responsible for processing input such as images or video into single frames and breaking it down into features that highlight visual patterns, like textures or distinct objects. Processed information is being handed over to the Projection Module, which maps the extracted features into raw text data. This step is crucial in the process of analyzing visual data in that it forms the connection between the encoded visual input and the vision LMM’s ability to generate coherent answers based off of the projection modules text output. The extracted output of the projection module is stored in an embedding space shared by the projection module and the vision LLM. Doing so allows the model to align visual information with text descriptions to further improve upon its abilities.  

Finally comes the vision LLM. Built on the transformer architecture, it is trained to analyze relationships between visual and text data, realized through self-attention mechanisms and feed-forward neural networks. Through the extracted information of the projection module, connecting visual features with textual data is enhancing its ability to reason and generate responses greatly. 

In the domain of text understanding, the core piece of an LLMs input understand comes from the use of transformers. Their operations range from Self-Attention Mechanisms to grasp a better understand of a phrases structure by weighing word groups and contextualizing distant words in relative comparison, which proves to be difficult otherwise. To Gradient Descent to adjust parameters and weights during the training phase, effectively reducing computing costs by a large margin. Transformers prove especially useful to produce coherent and relevant outputs. A prevalent example of such a transformer is Generative Pre-Trained Transformer 3 (GPT-3) by OpenAI which operates as a decoder-only transformer, not leveraging capabilities as Self-Attention or Encoder-Layers.

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