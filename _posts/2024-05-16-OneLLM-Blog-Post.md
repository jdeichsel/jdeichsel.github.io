---
title: 'A comprehensive introduction to OneLLM: One Framework to Align All Modalities with Language'
date: 2024-05-16
permalink: posts/onellm/
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
	- Architecture
3. Experiments
	- Quantitative & Qualitative Analysis
4. Ablation
5. Pros & Cons 
6. Conclusion


# LLM
Large Language Models (LLM) are artificial neural networks that use the transformer architecture. Which was introduced in 2017 by Vaswani et al. in the paper “Attention is all you need” [1]. Where they used it for translation purposes. 
Nowadays LLM excel at general natural language processing tasks like question answering, reasoning or prompt generating. The reason behind is the extraordinary parallelizing and scaling capabilities of transformers which allow them to be trained on large amounts of data in a relatively short time.


Transformer
======

Abstract
------

To explain how the transformer works we will consider translation from English to French sentences. We start on a highly abstract level and later go a little bit more in detail.
The Transformer consist of an encoder module, a decoder module
and the Attention Mechanism.
The encoder receives an input sequence, i.e. an English sentence as
vectors. It processes the data and gives an abstract continuous
representation that holds all learned information as an output vector.
The output is then fed into the decoder. During the training stage the
decoder additionally receives the final correct output as input,
e.g. French Translation of Input sentence. It learns to predict the
output step by step. During the application the output remains a
step-by-step prediction, which is now again fed into the decoder.
The driving force behind the transformer is the attention mechanism
within both encoder and decoder. It allows the model to focus on
different parts of the input sequence to weigh the importance
of the information.
To solve natural language processing tasks a model has to first understand the input sentence.  For example, “The server brings you your drinks.” and “I just crashed the server.” both use the word “server”, but one means waiter and the other computer, we humans understand the context, so we can distinguish them. The transformer achieves this, thanks to self-attention. The attention is turned to the input text itself. This helps in understanding the context of a word within a sentence.
In the transformer architecture they use multi-head self-attention. Essentially, they perform self-attention multiple times in parallel, so different heads can prioritize different parts of the input data and capture complex relationships within.
Here in Figure:[2] you can see a visualization of multi-head self-attention on the sentence “the cat sat on the mat” where one head weighs that for the word “sat” that “cat” is most important. Answering “what or who” sat, and the other head prioritize the word “on” and “mat” answering “where” it sat. you can also follow this link [https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ?usp=sharing] to test it out for yourself!

Detailed
------
Now in more detail:
The input sequence, e.g. text, is converted to numerical representations called tokens. Each token is converted into a vector through input embedding. The positional information of each word is also encoded in the vectors themselves, so that the word-order-information is stored in the data and not in the network-structure itself. This allows the model to learn the importance of word order from data, which makes it easier to train.  At each layer, each token is then contextualized with other tokens via a parallel multi-head attention mechanism. This allows the signal to amplify key tokens and lower less important tokens. This is done by the self-attention function.

The Input being the query vector Q, the key vector K and the value vector V. The query vector represents the word being attended to, while the key vectors represent all the words in the sentence. The value vectors store the information associated with each word. The key/value/query concept is analogous to retrieval systems. For example, when you search for videos on Youtube, the search engine will map your query (text in the search bar) against a set of keys (video title, description, etc.) associated with candidate videos in their database, then present you the best matched videos (values). [citation]
The attention weights are computed by taking the dot product between the query and key vectors, followed by a SoftMax operation to obtain a distribution over the words. And then multiplied with the value vector.
This is being processed multiple times. The outputs are then concatenated and linearly transformed to obtain the final representation. By using multiple attention heads, the model can capture both local and global dependencies, allowing for a more comprehensive understanding of the input sequence.

So, the most important information to remember: Transformers have incredible parallelizing capabilities so that they can be trained on massive amounts of data in a relatively short time. 
This makes them not only interesting for NLP tasks but also in other domains, like computer vision.


ViT
======
Moving on to a more visual part of LLMs, Dosovitskiy et al. [2] presented the Vision Transformer ViT, where they applied the transformer architecture for image processing. The input is a 2D image which is first divided into same size patches, which are then converted into 1D vectors and processed through the encoder part of the transformer. The extracted information holding vectors are then passed to a classification head a simple feed forward layer, which Classifies the input images this is the output.


# Vision-LLM
The progress in LLMs has also inspired researchers to employ LLMs as an interface for multimodal tasks, such as vision-language learning, audio and speech recognition or video understanding. The goal is to align different modalities like vision and audio with language so that a model can receive an image and a prompt as input and produce an answer. 
A vision LLM for example, typically consists of a visual encoder, like ViT, an LLM, and a projection module connecting the two components. The visual encoder extracts the most important information from an image. The projection module aligns image and text, so that the LLM can produce an output prompt. The vision LLM is first trained on massive amounts of paired image-text data to obtain a robust vision-language alignment and then fine-tuned on visual instruction datasets, enabling it to complete various instructions tied to visual inputs, like describing an image. 
For other modalities the architecture is pretty much the same, just with a modality specific encoder.


# MLLM
There are also several attempts to integrate multiple modalities into one Multimodal - LLM (MLLM). Most previous works such as ChatBridge [3,4] use a Vision-LLM as a basis and extent to new modalities by aligning each with the LLM using modality-specific encoders and projection modules. However, these modality-specific encoders usually differ in architecture and considerable effort is required to unify them into a single framework. Furthermore, pretrained encoders that deliver reliable performance are usually restricted to widely used modalities such as image, audio, and video. This limitation poses a constraint on MLLMs’ ability to expand to more modalities. 
Thus, a crucial challenge for MLLMs is how to build a unified and scalable encoder capable of handling a wide range of modalities.

# OneLLM
Okay, let’s take a brief look at the overall architecture of what OneLLM is working with.


The attentive reader will realize quickly that this structure differs in multiple pieces from the MLMMs that we’ve visited earlier, and it is! Most notably, the Encoder and Projection Module are now unified to work on any given modality, rather than having one for each modality. We’ll get later as to why this crucial to OneLLM’s effectiveness.
To take a deeper dive into each of these mechanisms, let’s first have a look at their basic functionalities.
Going through by the order of how inputs are processed, we’re starting with the Tokenizers.
There is nothing new about the general structure of inputting separate modalities compared to the MLLM’s we’ve revisited earlier. Since each modality is of a unique and very different data structure, processing them separately reduces headaches by a lot.
To give a better understanding of how the tokenizing works in this specific model, we want to show each step of the process by an example of the Visual Tokenizer.

Inputting Modalities 
======
Okay, we have all these nice different modalities, but how do we convert them into numbers so that the model can work with them? This is where the Tokenizers come in!
To successfully convert all our different inputs into tokens – which are essentially vectors – we will first need to quantify our input signals. Because of the established variety in modality structures, this is also being done depending on the type of modality that is being input.

As an example for the Visual Tokenizer, the image is input using the RGB channels. Our input is basically split up into 3 channels of different signals:


Convolution Layers
======
Here, another problem presents itself: the inputs are often far too large to be interpreted precisely,
at least in the current version of the model. This can be either too high-resolution images, too long videos, or too detailed point clouds.
To circumvent this, Convolution Layers are being applied to the input signal to efficiently break it down while still retaining positional data.
What this means is that we will be applying a separate matrix – the kernel – to our original input signal and calculate a weighted sum as a result. Most interesting here is that the result stays in the same relative position within the matrix.

What this also allows us to do is to interpret patterns within the original image. This highly depends on the usage of our kernel’s weights but can highlight certain features of images while still downgrading the overall complexity of the input.
Here is an example of applying a vertical edge detector kernel, the Horizontal Sobel Kernel, to an image. It’s called that way because it is looking for jumps in grey values on the horizontal plane, which basically highlights vertical objects!
Notice how vertical features are being highlighted while horizontal features almost disappear. 


Visual Tokenizer
======
Looking at how the Convolutional Layer for our Visual Tokenizer is defined, we can observe the input- and output channels as well as the kernels’ size, and another parameter S, which stands for Stride.


Striding simply refers to how far the kernel is moving over the input data horizontally and vertically for each new calculation. The animation that you’ve seen earlier has a Kernel Size of K = (3,3) and a Stride of S = (1,1).
As you can see, the kernel and stride size match!
Why is that interesting? Because the kernel won’t overlap any grey values from separate calculations with each other. And thus, we can create the smallest possible matrix with the minimal number of calculations without missing any values.  This is specifically important for visual inputs as OneLLM is largely being trained on visual datasets and has great accuracy on this modality.

Now, generally, visual inputs are denoted as tokens $$ x \in R^{H x W} $$ with H and W being the height and width of the image, respectively. However, since videos are also being processed, we will denote these as tokens $$ x \in R^{T x H x W} $$, where T is the number of frames of a video. 
Images are essentially a one-frame video input $$ x \in R^{1 x H x W} $$.
When parallel-feeding these tokens into our tokenizer, the output is going to be 
$$T x \frac{H}{14} x \frac{W}{14} $$ tokens ( $$ \frac{1}{14} $$ because of the kernel’s input reduction!).
test test

Universal Encoder 
======
The tokens are then fed into the Universal Encoder. The Universal Encoder is a frozen pretrained Vision LLM, in this case CLIP-ViT [5]. Which combines ViTs image processing capabilities and CLIPS robust image to text understanding. As previously mentioned this part of the model is already  pretrained on extensive image-text data, so it already posses robust vision and language alignment, which can then easily be transferred to other modalities. The Vision LLM extracts high dimensional features from the tokens xm. These are add to learnable modality tokens qm, which hold the information of the current modality type. This concatenation is then fed into the UPM.

Universal Projection Module 
======
Let’s get to one of the major players of OneLLM’s architecture, the Universal Projection Module, or UPM for short. 
There is two components that make up the UPM: the Projection Experts and the Modality Router.


Experts
------










