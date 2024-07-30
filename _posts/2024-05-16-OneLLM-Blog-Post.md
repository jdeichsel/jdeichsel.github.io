---
title: 'A comprehensive introduction to OneLLM: One Framework to Align All Modalities with Language'
date: 2024-05-16
permalink: posts/onellm/
tags:
  - OneLLM
  - language model
---

# Motivation

Deep Learning has revolutionized various fields like image recognition and language processing by handling complex data structures and finding patterns within those.\
However, as tasks become increasingly multimodal, integrating different data sources like text, images, audio, and video poses new challenges. Traditional models, typically focusing on fewer than three modalities, fall short of this transformation.\
[OneLLM](https://doi.org/10.48550/arXiv.2312.03700) by Han et al. [1] aims to create a unified framework to handle and align multiple modalities with language, addressing inefficiencies in existing models that rely on limited modality-specific encoders.\
This blogpost explores OneLLM's development, introducing a framework that aligns eight modalities with language using a unified multimodal encoder and a progressive alignment pipeline.\
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
Large Language Models (LLM) are artificial neural networks that use the transformer architecture which was introduced in 2017 by Vaswani et al. in the paper [Attention is all you need](https://doi.org/10.48550/arXiv.1706.03762) [6]. Where they used it for translation purposes.\
Nowadays LLM excel at general natural language processing tasks like question answering, reasoning or prompt generating.\
The reason behind is the extraordinary parallelizing and scaling capabilities of transformers which allow them to be trained on large amounts of data in a relatively short time.


Transformer
======

Abstract
------

To explain how the transformer works we will consider translation from English to French sentences. We start on a highly abstract level and later go a little bit more in detail.\
The Transformer consist of an encoder module, a decoder module and the Attention Mechanism.


![](/images/transformer_architecture.png)\
*Figure 1: Transformer architecture by Vaswani et al. in [Attention is all you need](https://doi.org/10.48550/arXiv.1706.03762) [6]*


The encoder receives an input sequence, i.e. an English sentence as
vectors. It processes the data and gives an abstract continuous
representation that holds all learned information as an output vector.
The output is then fed into the decoder. During the training stage the
decoder additionally receives the final correct output as input,
e.g. French Translation of Input sentence. It learns to predict the
output step by step. During the application the output remains a
step-by-step prediction, which is now again fed into the decoder.\

The driving force behind the transformer is the attention mechanism
within both encoder and decoder. It allows the model to focus on
different parts of the input sequence to weigh the importance
of the information.\

To solve natural language processing tasks a model has to first understand the input sentence.  For example, “The server brings you your drinks.” and “I just crashed the server.” both use the word “server”, but one means waiter and the other computer, we humans understand the context, so we can distinguish them. The transformer achieves this, thanks to self-attention. The attention is turned to the input text itself. This helps in understanding the context of a word within a sentence.\

In the transformer architecture they use multi-head self-attention. Essentially, they perform self-attention multiple times in parallel, so different heads can prioritize different parts of the input data and capture complex relationships within.
Here in Figure 2 you can see a visualization of multi-head self-attention on the sentence “the cat sat on the mat” where one head weighs that for the word “sat” that “cat” is most important. Answering “what or who” sat, and the other head prioritize the word “on” and “mat” answering “where” it sat. You can also follow this [link](https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ?usp=sharing) to test it out for yourself!


![](/images/cat_mat_example.png)\
*Figure 2: Example of applying multi-head self-attention [12]*

Detailed
------
Now in more detail:
The input sequence, e.g. text, is converted to numerical representations called tokens. Each token is converted into a vector through input embedding. The positional information of each word is also encoded in the vectors themselves, so that the word-order-information is stored in the data and not in the network-structure itself.\
This allows the model to learn the importance of word order from data, which makes it easier to train.  At each layer, each token is then contextualized with other tokens via a parallel multi-head attention mechanism. This allows the signal to amplify key tokens and lower less important tokens. This is done by the self-attention function.

$$ Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}}) $$\

The Input being the query vector Q, the key vector K and the value vector V. The query vector represents the word being attended to, while the key vectors represent all the words in the sentence. The value vectors store the information associated with each word. The key/value/query concept is analogous to retrieval systems. For example, when you search for videos on Youtube, the search engine will map your query (text in the search bar) against a set of keys (video title, description, etc.) associated with candidate videos in their database, then present you the best matched videos (values).\
The attention weights are computed by taking the dot product between the query and key vectors, followed by a SoftMax operation to obtain a distribution over the words. And then multiplied with the value vector.\
This is being processed multiple times. The outputs are then concatenated and linearly transformed to obtain the final representation. By using multiple attention heads, the model can capture both local and global dependencies, allowing for a more comprehensive understanding of the input sequence.

![](/images/transformer_example.png)\
*Figure 3: Transformer architecture in further detail, showcasing the implementation of attention mechanisms [11]*

So, the most important information to remember: Transformers have incredible parallelizing capabilities so that they can be trained on massive amounts of data in a relatively short time. \
This makes them not only interesting for NLP tasks but also in other domains, like computer vision.


ViT
======
Moving on to a more visual part of LLMs, Dosovitskiy et al. in [An Image is worth 16x16 words: Transformers for Image Recognition at Scale](https://doi.org/10.48550/arXiv.2010.11929) [8] presented the Vision Transformer ViT, where they applied the transformer architecture for image processing. The input is a 2D image which is first divided into same size patches, which are then converted into 1D vectors and processed through the encoder part of the transformer. The vectors carrying the key information are then passed to a classification head - a simple feed forward layer - which outputs a classification of the images.

![](/images/ViT_example.png)\
*Figure 4: Vision-transformer (ViT) architecture, with transformer encoder in greater detail by Dosovitskiy et al. [8]*

# Vision-LLM
The progress in LLMs has also inspired researchers to employ LLMs as an interface for multimodal tasks, such as vision-language learning, audio and speech recognition or video understanding. The goal is to align different modalities like vision and audio with language so that a model can receive an image and a prompt as input and produce an answer. \
A vision LLM for example, typically consists of a visual encoder, like ViT, an LLM, and a projection module connecting the two components. The visual encoder extracts the most important information from an image. The projection module aligns image and text, so that the LLM can produce an output prompt. The vision LLM is first trained on massive amounts of paired image-text data to obtain a robust vision-language alignment and then fine-tuned on visual instruction datasets, enabling it to complete various instructions tied to visual inputs, like describing an image. \
For other modalities the architecture is pretty much the same, just with a modality specific encoder.

![](/images/vision_LLM_architecture.png)\
*Figure 5: Example: vision LLM by Li et al. [BLIP-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and Large Language Models](https://doi.org/10.48550/arXiv.2301.12597) [9]*


# MLLM
There are also several attempts to integrate multiple modalities into one Multimodal - LLM (MLLM). Most previous works such as [ChatBridge](https://doi.org/10.48550/arXiv.2305.16103) [5] use a Vision-LLM as a basis and extent to new modalities by aligning each with the LLM using modality-specific encoders and projection modules. However, these modality-specific encoders usually differ in architecture and considerable effort is required to unify them into a single framework. Furthermore, pretrained encoders that deliver reliable performance are usually restricted to widely used modalities such as image, audio, and video. This limitation poses a constraint on MLLMs’ ability to expand to more modalities. \
Thus, a crucial challenge for MLLMs is how to build a unified and scalable encoder capable of handling a wide range of modalities.


![](/images/MLLM_architecture.png)\
*Figure 6: ChatBridge Architecture by Zhao et al. [5]*


# OneLLM
Okay, let’s take a brief look at the overall architecture of what OneLLM is working with.

![](/images/onellm_total_architecture.png)\
*Figure 7: Overview of OneLLM’s architecture by Han et al. [1]*

The attentive reader will realize quickly that this structure differs in multiple pieces from the MLMMs that we’ve visited earlier, and it is! Most notably, the Encoder and Projection Module are now unified to work on any given modality, rather than having one for each modality. We’ll get later as to why this crucial to OneLLM’s effectiveness.\
To take a deeper dive into each of these mechanisms, let’s first have a look at their basic functionalities.\
Going through by the order of how inputs are processed, we’re starting with the Tokenizers.
There is nothing new about the general structure of inputting separate modalities compared to the MLLM’s we’ve revisited earlier. Since each modality is of a unique and very different data structure, processing them separately reduces headaches by a lot.\
To give a better understanding of how the tokenizing works in this specific model, we want to show each step of the process by an example of the Visual Tokenizer.

Inputting Modalities 
======
Okay, we have all these nice different modalities, but how do we convert them into numbers so that the model can work with them? This is where the Tokenizers come in!\
To successfully convert all our different inputs into tokens – which are essentially vectors – we will first need to quantify our input signals. Because of the established variety in modality structures, this is also being done depending on the type of modality that is being input.\
As an example for the Visual Tokenizer, the image is input using the RGB channels. Our input is basically split up into 3 channels of different signals:

![](/images/conv_layer_cin.png)\
*Figure 8: Visual Tokenizer used in OneLLM [1]*\
<img src="/images/three_d_array.png" width="416" height="400" />\
*Figure 9: [How to convert an RGB image to Grayscale](https://e2eml.school/convert_rgb_to_grayscale) [10]*


Convolution Layers
======
Here, another problem presents itself: the inputs are often far too large to be interpreted precisely,
at least in the current version of the model. This can be either too high-resolution images, too long videos, or too detailed point clouds.\
To circumvent this, Convolution Layers are being applied to the input signal to efficiently break it down while still retaining positional data.\
What this means is that we will be applying a separate matrix – the kernel – to our original input signal and calculate a weighted sum as a result. Most interesting here is that the result stays in the same relative position within the matrix.

![](/images/conv_layer_gif.gif)\
*Figure 10: Applying Convolution Layers to an input signal from [A guide to convolution arithmetic for deep learning](https://doi.org/10.48550/arXiv.1603.07285) [3]*

What this also allows us to do is to interpret patterns within the original image. This highly depends on the usage of our kernel’s weights but can highlight certain features of images while still downgrading the overall complexity of the input.
Here is an example of applying a vertical edge detector kernel, the Horizontal Sobel Kernel, to an image. It’s called that way because it is looking for jumps in grey values on the horizontal plane, which basically highlights vertical objects!
Notice how vertical features are being highlighted while horizontal features almost disappear. 

![](/images/conv_layer_example.png)\
*Figure 11: Example: Horizontal Sobel Kernel detecting vertical features from [Intuitively Understanding Convolutions for Deep Learning](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1) [4]*

Visual Tokenizer
======
Looking at how the Convolutional Layer for our Visual Tokenizer is defined, we can observe the input- and output channels as well as the kernels’ size, and another parameter S, which stands for Stride.

![](/images/conv_layer_KS.png)\
*Figure 8: Visual Tokenizer used in OneLLM [1]*

Striding simply refers to how far the kernel is moving over the input data horizontally and vertically for each new calculation. The animation that you’ve seen earlier has a Kernel Size of K = (3,3) and a Stride of S = (1,1).\
As you can see, the kernel and stride size match!
Why is that interesting? Because the kernel won’t overlap any grey values from separate calculations with each other. And thus, we can create the smallest possible matrix with the minimal number of calculations without missing any values.  This is specifically important for visual inputs as OneLLM is largely being trained on visual datasets and has great accuracy on this modality.\
Now, generally, visual inputs are denoted as tokens $$ x \in R^{H \times W} $$ with H and W being the height and width of the image, respectively. However, since videos are also being processed, we will denote these as tokens $$ x \in R^{T \times H \times W} $$, where T is the number of frames of a video. 
Images are essentially a one-frame video input $$ x \in R^{1 \times H \times W} $$.\
When parallel-feeding these tokens into our tokenizer, the output is going to be 
$$T \times \frac{H}{14} \times \frac{W}{14} $$ tokens ( $$ \frac{1}{14} $$ because of the kernel’s input reduction!).

Universal Encoder 
======
The tokens are then fed into the Universal Encoder.\
The Universal Encoder is a frozen pretrained Vision LLM, in this case CLIP-ViT [13] which combines ViTs image processing capabilities and CLIPS robust image to text understanding. As previously mentioned, this part of the model is already pretrained on extensive image-text data, so it already posseses robust vision and language alignment, which can then easily be transferred to other modalities. The Vision LLM extracts high dimensional features from the tokens $$x_m$$. These are added to learnable modality tokens $$q_m$$, which hold the information of the current modality type.\
This concatenation is then fed into the UPM.

Universal Projection Module 
======
Let’s get to one of the major players of OneLLM’s architecture, the Universal Projection Module, or UPM for short. \
There are two components that make up the UPM: the Projection Experts and the Modality Router.

<img src="/images/UPM_architcture.png" width="431" height="400" />\
*Figure 12: Overview of OneLLM's Universal Projection Module (UPM) [1]*

Experts
======
As the name suggests, this model is not deploying modality-specific experts as we’ve seen before, but rather projection experts that apply to any modality.\
Essentially, the experts themselves are a stack of transformer layers – as we’ve discussed earlier – and are pre-trained on image-text data. When inputting our combined token, these experts apply their own weights in parallel!\
This looks like the following: consider an input $$ UPM \left ( \left [ q_m, x_m \right ] \right ) $$, which is a concatenated list of tokens from the learnable Modality Token q and the tokenized original input x, to which we apply each expert K. Applying the experts yields K outputs $$ P_K \left ( \left [q_m, x_m \right ] \right ) $$. Note that m is the current modality of the input.\
Imagine in a real-world setting an expert voicing his opinion on something that concerns his field of expertise.


Router
======
Now that we have all these different experts with their opinions, we need a way to quantify the importance of each of their outputs. This is where the Modality Router comes into play! \
We’re going to be applying the method of a soft router in OneLLM. We will cover other options of routers in the Ablation section.\
The Soft Router from [From Sparse to Soft Mixture of Experts](https://doi.org/10.48550/arXiv.1603.07285) [2] essentially is a straightforward Multi-Layer Perceptron. Meaning that it is a type of neural network used to analyze subsections of the input data and to consider their overall importance in the context of the entire input data.

![](/images/soft_moe.png)\
*Figure 13: Assigning weighted mixtures of the images’ sub-sections to experts [2]*

In the example here, you can see that instead of allocating one piece of the image to each expert to analyze, it is assigning a weighted average over each column to the experts. The images’ weights are previously applied by the router by order of importance. \
To now compare each weight with one another, a SoftMax activation function is applied. The SoftMax in particular is very helpful – even for the human eye! – to solve classification problems. As the sum of probabilities that are output for all items is always 1.\
Therefore, we denote the routing weight for each expert as\
$$ w_m=\sigma \circ \mathds{R}_m\left(\left[q_m,x_m\right]\right) $$

<img src="/images/softmax_example.png" width="596" height="300" />\
*Figure 14: Example of applying softmax classification problem from [Softmax Activation Function: Everything You Need to Know](https://www.pinecone.io/learn/softmax-activation/) [14]. Note that the output sums to 1*

Now, to apply this weight to the respective experts and to obtain a final output, we’re going to be taking a weighted average over each weight and their experts. Our result $$\left[q_m,x_m\right]$$ is going to look as the following:\
$$\left[{\bar{q}}_m,{\bar{x}}_m\right]=UPM\left(\left[q_m,x_m\right]\right)\ =\ \sum_{k=1}^{K}{w_m\ast P_k\left(\left[q_m,x_m\right]\right)}$$\
Great! We’ve applied our general Projection Experts to our input, leveraged the weights of each subsection of the input data and weighted each expert’s contribution to generate modified versions of the Modality Token and the original input.\
Going forward, the input is no longer of interest. From here on out we will drop the input data in favor of the Modality Token, which we’ll be using to input into the Large Language Model. This is due to the uniform dimensions the modality tokens are set in, as well as the unpredictable dimensions of each modality’s input. Doing so allows us to more efficiently compute over the LLM’s expected input without needing to expect different structures or input forms.


# Modality Alignment / Instruction Tuning
So, that’s it right? We’ve successfully turned an input of any modality into tokens, computed them over our experts and used the UPM to embed the modality token into the LLM to generate a response, given a user’s prompt.\
Not quite! The model hasn’t yet been trained at all. Before we get to process our modalities, we first need to teach OneLLM to correctly identify and analyze inputs and reasoning, conversation, and so on.\
This is where Modality Alignment and Instruction Tuning come into play. 


Modality Alignment
------
First, let’s have a look at Modality Alignment.\
You might assume that, similar to Large Language Models, a training dataset with sufficient size and items of every modality will do the trick in successfully training and aligning our modalities.\
However, that doesn't apply here. This is mostly due to the dataset imbalance, as the amount of moderated items for each modality differs wildly. To put this into perspective, this chart showcases some of the more quantifiable modalities in number of items per training dataset.

![](/images/modalities_size.png)\
*Figure 15: Comparison of quantifiable modality datasets used to train OneLLM [1]*

As is apparent here, the image dataset is off the charts with over 1 billion items!
While this looks concerning, we can assure you that it is. With such high differences in training datasets, extreme cases of bias can emerge during training where the model performs much more accurately to image inputs than to any other modalities. \
To combat this behavior, we’re going to be using a strategy called Modality Alignment. As the name suggests, we will progressively align all modalities into OneLLM, no matter the number of modalities.\
The goal here is to train the Modality Tokenizers and the UPM, while keeping the Large Language Model frozen, i.e. not learning. 

![](/images/modality_alignment.png)\
*Figure 16: Showcasing Modality Alignment training phase (red) by freezing certain parts of the architecture (blue) [1]*

Starting off at the image training dataset, we’re going to employ a pre-trained vision LLM which you’ve seen earlier, together with an Image Tokenizer and Image Projection Module and a single Image Expert. As it is now, we’re basically working with a vision LLM.\
After training on the image dataset, new modalities are continually being added with each training dataset. For that, we denote for timestep $$t$$ so that $$ \mathcal{M}_1\cup\mathcal{M}_2\cup\ldots\cup\mathcal{M}_{t-1} $$\
Essentially, all previous modalities have successfully been trained upon.\
One important issue when training models is catastrophic forgetting where the model simply forgets previously trained knowledge when working with increasingly large datasets.
To counteract on that, we will sample an equal amount of already trained items with the new dataset, showing the importance of sampling datasets by order of size.\
Alright, that’s it! We’ve trained our Tokenizers and UPM.\
Now, we’re looking at what is basically a captioning model. What is missing now is reasoning, conversation, etc. \
To gain these abilities, we’ll be training the LLM next in the Instruction Tuning Stage.


Instruction Tuning
------
The process of Instruction Tuning is rather straightforward. We essentially flip around the Modality Alignment process and now train the Large Language Model while keeping the Tokenizers and UPM frozen, i.e. not learning.

![](/images/instruction_tuning.png)\
*Figure 17: Showcasing Instruction Tuning training phase (red) by freezing certain parts of the architecture (blue) [1]*

To do so, there is a 2M items dataset specifically curated for OneLLM, containing items on all eight current modalities.\
And with the LLM training out of the way, we’re done! The model is now fully functional and can process all modalities, as well as answer user’s prompts in reasoning, question-answering or conversations.


# Experiments

Qualitative Analysis
======
To start we will look at the qualitative Analysis. As we know OneLLM can understand up to eight different modalities, here we show you how well it performs. We give it different modality specific inputs and a question. As you can see the model recognizes all modalities effectively like the point-cloud humanoid (d) or the fMRI signal obtained of a school bus and sunny weather (f).\
It can perform creative tasks, such as writing a poem based on beach sounds it heard (c).\
The model gives reasoning, e.g. what should you do if a bear approaches you (e). OneLLM is also capable of identifying activities in a depth/normal map (b). As well as recognizing real-world events and tie them into answers, for instance, it recognizes the movie poster of Oppenheimer and it knows that Oppenheimer was the inventor of the atomic bomb.

<img src="/images/qualitative_1.png" width="250" height="267" />
<img src="/images/qualitative_2.png" width="250" height="267" />\
*Examples a) and b)*

<img src="/images/qualitative_3.png" width="250" height="267" />
<img src="/images/qualitative_4.png" width="250" height="267" />\
*Examples c) and d)*

<img src="/images/qualitative_5.png" width="250" height="267" />
<img src="/images/qualitative_6.png" width="250" height="267" />\
*Examples e) and f)*

<img src="/images/qualitative_7.png" width="250" height="267" />
<img src="/images/qualitative_8.png" width="250" height="267" />\
*Examples g) and h)*\
*Figure 18 [1]*

Quantitative Analysis
======
Now we have a fully trained model and it's ready to go! But how good is it exactly? 
To answer this question, a few benchmarks have been provided to give a better understanding of how OneLLM is performing against similar MLLM’s. More interestingly are the specified LLM’s that are also provided within the benchmarks. We’ll get to why that is interesting in just a bit.
To start things off, we’ll have a look at the Image-Text Benchmarks first. Contrary to the Visual Tokenizer, this does not encompass all visual tasks but only images.

![](/images/image_text_benchmark.png)\
*Figure 19: Image-Text-Benchmark comparing MLLMs as well as LLMs. Scores in bold and underline represent the best and second best results, respectively within the MLLM category. Green scores represent the three best results per dataset [1]*

The results in bold and underline writing represent the best and second-best results in the MLLM category, respectively. \
Each of these models was – disregarding a few missing entries – trained on all of these datasets, which test each model’s capability in Visual Question-Answering (VQA) and Image Captioning.
Looking closely at the results, we can observe OneLLM being the strongest in its field, outperforming all related models on these datasets. Especially compared to [AnyMAL-70B](https://arxiv.org/pdf/2309.16058) [15] by Moon et al. which is working on approximately 10x as many parameters as OneLLM-7B, showing the extreme parameter and performance efficiency of OneLLM.\
However, if we look at the scores marked in green, the benchmarks show a little different perspective. Here, we marked the overall three best models from both categories. 
While OneLLM does still make it into the best three scores for half the datasets, it is nowhere near as dominant as when looking into the MLLM category only. This also goes for the other MLLM models, who are almost unable to secure a competitive score rating. 

Two more benchmarks we’d like to quickly brush over are the Video-Text (left) and Video-Audio-Text (right) benchmarks. 

![](/images/image_text_benchmark.png)\
*Figure 20: Example: Video-Text Benchmark (left) and Video-Audio-Text Benchmark (right) including comparison of zero-shot capabilities [1]*

While the previously established pattern largely repeats here with LLM’s being more accurate, OneLLM starts off on a disadvantage:\
It did not receive any training datasets concerning Video Question-Answering or Video-Audio Question-Answering. However, their related MLLMs [AnyMal-13B](https://arxiv.org/pdf/2309.16058) [15] and [ChatBridge-13B](https://doi.org/10.48550/arXiv.2305.16103) [5] did! 
This again goes to show just how strong the modalities are aligned with one another within OneLLM and can learn from each other, given the reduced number of parameters in comparison.


# Ablation
One last part we’d like to touch on is the Ablation, showcasing different implementations and their impact on performance.

![](/images/image_text_benchmark.png)\
*Figure 21: Impact of different implementation methods on performance [1]*

Most notably here is the Separate vs. Joint Training Mode. Quantifying the difference in performance between Separate Training – what regular MLLM’s do, training each modality separately – and OneLLM’s Joint Training turns into a massive performance loss in accuracy. \
The same goes for the Weight Initialization as to which modality should be trained first to then align all others to it. As expected, starting off with the largest available dataset is much better than choosing at random.


# Pros & Cons
This model is indeed a strong alternative and proof-of-concept in comparison to previous MLLM works, showcasing an impressive ability to work with relatively small modality datasets and to be able to align them quite effectively.\
The implementation of universal modules – such as the Universal Projection Module and Encoder – proved to be quite successful and paves the way for future works building upon the architecture.\
However, OneLLM is not without its drawbacks. While it does manage to unify eight different modalities into a single framework, there aren’t that many use-cases to justify doing such a task, specifically when upping the number of modalities within the framework.\
Additionally, one must decide whether taking up a model with multiple modalities is worth taking over multiple LLM with specialized use-cases and higher accuracy. \
While both methods have their own benefits, you’ll essentially end up leveraging OneLLM’s flexibility with an LLM’s accuracy – and not that much accuracy at that! 


# Conclusion
Thank you for your attention! We hope that you’ve gained some valuable insights into the inner working on OneLLM and what discerns it from other models!\
With that being said, we are very curious to see where Han et al. will be taking this project to, as future work announced changes to the granularity of analyzing data as well as providing more fine-tuned datasets. 


# References
[1] Han et al., 2024, [OneLLM: One Framework to Align All Modalities with Language](https://doi.org/10.48550/arXiv.2312.03700)\
[2] Puigcerver et al., 2024, [From Sparse to Soft Mixture of Experts](https://doi.org/10.48550/arXiv.1603.07285)\
[3] Dumoulin & Visin., 2018, [A guide to convolution arithmetic for deep learning](https://doi.org/10.48550/arXiv.1603.07285)\
[4] Irhum Shafkat, 2018, [Intuitively Understanding Convolutions for Deep Learning](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)\
[5] Zhao et al., 2020, [ChatBridge: Bridging Modalities with Large Language Model as a Language Catalyst](https://doi.org/10.48550/arXiv.2305.16103)\
[6] Vaswani et al., 2017, [Attention is all you need](https://doi.org/10.48550/arXiv.1706.03762)\
[7] Jesse Vig, 2019, [Deconstructing BERT, Part 2: Visualizing the Inner Works of Attention](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)\
[8] Dosovitskiy et al., 2021, [An Image is worth 16x16 words: Transformers for Image Recognition at Scale](https://doi.org/10.48550/arXiv.2010.11929)\
[9] Li et al., 2023, [BLIP-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and Large Language Models](https://doi.org/10.48550/arXiv.2301.12597)\
[10] Brandon Rohrer, 2019, [How to convert an RGB image to Grayscale](https://e2eml.school/convert_rgb_to_grayscale)\
[11] Lillian Weng, 2018, [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)\
[12] [BertViz Interactive Tutorials](https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ?usp=sharing)\
[13] Radford et al., 2021, [Learning transferable visual models from natural language super16 vision](https://arxiv.org/pdf/2103.00020)\
[14] Bala Priya C., 2023, [Softmax Activation Function: Everything You Need to Know](https://www.pinecone.io/learn/softmax-activation/)\
[15] Moon et al., 2023, [AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model](https://arxiv.org/pdf/2309.16058)\


























