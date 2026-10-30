# 07 - Neural Architecture Search I

- [Recap: Primitive Operations](#recap-primitive-operations)
- [Classic Building Blocks](#classic-building-blocks)
  - [Residual Bottleneck](#residual-bottleneck)
  - [ResNeXt](#resnext)
  - [MobileNet](#mobilenet)
  - [MobileNetV2](#mobilenetv2)
  - [ShuffleNet](#shufflenet)
  - [Multi-Head Self-Attention](#multi-head-self-attention)
- [Introduction to Neural Architecture Search (NAS)](#introduction-to-neural-architecture-search-nas)
  - [What is NAS?](#what-is-nas)
  - [Search Space](#search-space)
    - [Cell-Level Search Space](#cell-level-search-space)
    - [Network-Level Search Space](#network-level-search-space)
  - [Design The Search Space](#design-the-search-space)
  - [Search Strategy](#search-strategy)
    - [Grid Search](#grid-search)
    - [Random Search](#random-search)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Gradient Descent (DARTS)](#gradient-descent-darts)
    - [Evolutionary Search](#evolutionary-search)


When optimizing a model, we will never truly reach an optimium across all properties like storage, latency, energy demands (and second order impact like cooling demands) and accuracy. *There's trade-offs to be made: Efficiency vs. Accuracy.*

**Rather than compressing an existing model, can we design a small, compact and efficient model to begin with?**

## Recap: Primitive Operations

**For the MLP,** we find that $Y=WX+b$ concerns the shapes
- $X$: $(n,\ c_i)$ meaning an *input matrix* of shape batch $\times$ input channels
- $Y$: $(n,\ c_o)$ meaning a *results matrix* of shape batch $\times$ output channels
- $W$: $(c_o,\ c_i)$ meaning the *learnable weight matrix* of shape output channels $\times$ input channels
- $b$: $(c_o)$ meaning the bias vector with one bias value per output

>[!NOTE]
> With a batch of $1$, i.e. $n=1$, we find that the MACs accumulate as $c_o \cdot c_i$.

<img src="./attachments/Pasted%20image%2020260704192234.png" width="400" />

**For Convolutions** we have additional dimensions:
- $w_i,h_i$: *Width and height of the input feature map*
- $w_o,h_o$: *Width and height of the output feature map*
- $k_w, k_h$: *Width and height of the kernel*

|      | **1D Conv**     | **2D Conv**         |
| ---- | --------------- | ------------------- |
| $X$: | $(n,c_i,w_i)$   | $(n,c_i,h_i,w_i)$   |
| $Y$: | $(n,c_o,w_o)$   | $(n,c_o,h_o,w_o)$   |
| $W$: | $(c_o,c_i,k_w)$ | $(c_o,c_i,k_h,k_w)$ |
| $b$: | $(c_o)$         | $(c_o)$             |

For *1D Conv*, this looks like so:<br>
<img src="./attachments/Pasted%20image%2020260704201611.png" width="500" /><br>
*Source:* [Prakhar Ganesh, towardsdatascience.com, 2019](https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37/)

For *2D Conv*, this looks like so:<br>
<img src="./attachments/Pasted%20image%2020260704195156.png" width="500" />

>[!NOTE]
> With a batch of $1$, i.e. $n=1$, we find that the MACs accumulate as $c_o \cdot c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o$.

**For Grouped Convolutions** we have an additional term $g$ denoting how many 'partial' kernels the processing is distributed over.<br>
*Apart from the kernel weights $W$, the part-taking tensor shapes remain as with 2D Conv.*

|      | **Grouped Conv**                       |
| ---- | -------------------------------------- |
| $W$: | $(g \cdot c_o / g, c_i / g, k_h, k_w)$ |

<img src="./attachments/Pasted%20image%2020260704202415.png" width="450" /><br>
*Source:* [\[Gibson et al., 2020\]](https://www.research.ed.ac.uk/en/publications/optimizing-grouped-convolutions-on-edge-devices/)

>[!NOTE]
> With a batch of $1$, i.e. $n=1$, we find that the MACs now accumulate as $c_o \cdot c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o / g$.

**For Depth-Wise Convolutions**, where the number of groups of the kernel equals the number of channels in the input, we again only see an *altered kernel* $W$ *from the 2D Conv, but it now also affects* $Y$:

|      | **Depth-Wise Conv** |
| ---- | ------------------- |
| $W$: | $(c_i, k_h, k_w)$   |
| $Y$: | $(n,c_i,h_o, w_o)$  |

Each input channel has its own kernel, which produces one output feature map. Consequently, the output maintains the same number of channels as the input. *Each channel only ever affects its own designated kernel. Information does not flow between input channels.*

<img src="./attachments/Pasted%20image%2020260704205021.png" width="300" /><br>
*Source:* [\[Abdallah et al., 2024\]](https://www.researchgate.net/publication/381306183_DepthGen_Lite_Efficient_Inpainting_with_Dual-Gen_and_Depthwise_Convolutions)

With a batch of $1$, i.e. $n=1$, we find that the MACs accumulate as $c_o \cdot k_h \cdot k_w \cdot h_o \cdot w_o$.

**For the 1x1 Convolution** we find that $k_h = k_w = 1$. The input channels get flattened into a single dimension by $c_o$ many kernels. *This gives us full control over the output channel dimension.*

<img src="./attachments/Pasted%20image%2020260704205408.png" width="700" /><br>
*Source:* [community.deeplearning.ai](https://community.deeplearning.ai/t/difference-between-1-1-convolution-and-pointwise-convolution/149338)

With a batch of $1$, i.e. $n=1$, we find that the MACs accumulate as $c_o \cdot c_i \cdot h_o \cdot w_o$.

## Classic Building Blocks

Using these operations, e.g. affine transformation, 1D and 2D Conv, etc., we can build blocks out of which we can construct networks.

### Residual Bottleneck

One such block is the **residual bottleneck block:**<br>
<img src="./attachments/Pasted%20image%2020260704212148.png" width="150" /><br>
*Source:* [\[Srinivas et al., 2021\]](https://arxiv.org/abs/2101.11605)

In a deep convolutional neural network, this block *reduces computation by first compressing the number of channels before applying the expensive* $3 \times 3$ *convolution.* The network then expands the channels back to their original dimension, *encouraging more efficient feature learning.* Channels may for example be reduced from 2048 to 512 using $1 \times 1$ convolutions for both reduction and expansion. A residual connection around the block helps preserve gradient flow.

Even though this block consists of more components, the sum of said components is easier on compute than just the single full-sized $3\times 3,\ 2048$ layer.

**At batch size $1$, we find:**
- MACs for the $3 \times 3,\ 2048$ layer: $c_o \cdot c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o = 2048 \cdot 2048 \cdot 3 \cdot 3 \cdot h_o \cdot w_o = \underline{37748736} \cdot h_o \cdot w_o$
- MACs for the residual bottleneck of a $3\times 3,\ 512$ layer with two $1\times 1$ convolutions: $\underline{4456448} \cdot h_o \cdot w_o$
*This is a $\sim 8.5 \times$ reduction.*

### ResNeXt

Advancing beyond the *residual bottleneck,* we can have the $3 \times 3$ convolution be a grouped convolution:<br>
<img src="./attachments/Pasted%20image%2020260705142643.png" width="750" /><br>
*Source:* [\[Xie et al., 2017\]](https://arxiv.org/abs/1611.05431)

If we have $32$ groups, each group has to process $128 / 32 = 4$ channels from the input in a respective $3\times 3$ convolution. This gets concatenated alog the channel dimension.
We can further adapt the last $128,\ 1\times 1,\ 256$ convolution to also be done per group, and then sum up these group-wise $1\times 1$ convolutions for the final output, saving yet more parameters.

- MACs for $3\times 3,\ 256$ layer: $256\cdot 256 \cdot 3 \cdot 3 \cdot h_o \cdot w_o =  \underline {589824} \cdot h_o \cdot w_o$
- MACs for the third configuration:  $32 \cdot \left( (2 \cdot 256 \cdot 4) + 4 \cdot 4 \cdot 3 \cdot 3 \right) \cdot h_o \cdot w_o = \underline{70144} \cdot h_o \cdot w_o$
*This is a $\sim 8.41\times$ reduction.*

### MobileNet

We already mentioned the depthwise convolution and the 1x1 convolution.<br>
Both combined form what is called a *depthwise-separable block, prominently used in MobileNet:*
- Use depthwise convolution to capture spatial information (per channel information)
- Use 1x1 (pointwise) convolution to exchange information across channels

<img src="./attachments/Pasted%20image%2020260705151330.png" width="350" /><br>
*Source:* [\[Howard et al., 2017\]](https://arxiv.org/abs/1704.04861)

### MobileNetV2

Incremental to the *depthwise-seperable block* from MobileNet, MobileNetV2 utilizes the *inverted bottleneck block:*
- **Problem:** Depthwise convolution provides only little capacity/capability for the model
- **Solution:** Increase the depthwise convolution's input and output channels to improve its limited capacity; Depthwise convolution's cost only grows linearly with input channels, therefore doing this is still affordable

<img src="./attachments/Pasted%20image%2020260705154311.png" width="500" /><br>
*Source:* [\[Sandler et al., 2018\]](https://arxiv.org/abs/1801.04381)

Increasing the channel count poses a drawback, though.<br>
The activation memory demand grows by the multiple by which we increase the channels.

<img src="./attachments/Pasted%20image%2020260705154845.png" width="500" /><br>
*Source:* [\[Sandler et al., 2018\]](https://arxiv.org/abs/1801.04381)

>[!NOTE]
> MobileNet is very good for efficient inference, but it is not good for efficient training.
> MobileNetV2 in particular, while more capable, becomes quite costly for training runs with large batch sizes.
> FLOPs are reduced, activation memory, the main bottleneck during training, is not reduced, but increased, compared to ResNet-18.

### ShuffleNet

*Compared to MobileNet, we now replace the* $1\times 1$ *convolutions with grouped* $1\times 1$ *convolutions.*<br>
To still *maintain some proper information flow across all the channels,* which groups natually would not allow, we just *shuffle the channels for each ShuffleNet layer to have groups process different sets of channels every time,* this way exchanging information across different channel sub-groups every time.

<img src="./attachments/Pasted%20image%2020260705160604.png" width="700" /><br>
*Source:* [\[Zhang et al., 2018\]](https://arxiv.org/abs/1707.01083)

### Multi-Head Self-Attention

First time the transformer architecture is indirectly referenced, the MHSA projects the inputs $Q$, $K$ and $V$ through $h$ different learned linear projections and perform scaled dot-product attention on each of these versions of $Q$, $K$, and $V$ in parallel. Outputs get concatenated (literally appended) and then projected into the target shape.

$$\text{head}_i = \text{Attention}(Q\underline{W_i^Q},\ K\underline{W_i^K},\ V\underline{W_i^V})$$

$$\text{MultiHead}(Q,\ K,\ V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_o$$

<img src="./attachments/Pasted%20image%2020260705165202.png" width="700" /><br>
*Source:* [\[Vaswani et al., 2017\]](https://arxiv.org/abs/1706.03762)

>[!NOTE]
> With $N$ being the context size, i.e. the token count, if this gets larger, MHSA gets quite a lot more expensive and very very quickly so.

Suppose for the following example that every word is one token. Feeding the sentence `"I bet the video game is a lot more fun than the film"` has different words/tokens attend differently to other words/tokens:<br>

<img src="./attachments/Pasted%20image%2020260705164858.png" width="300" />

We can see that token `the` attends to pretty much nothing (see the column), and so does `a`.<br>
We will discuss what to do with such tokens later, as they offer room for efficiency improvements.

## Introduction to Neural Architecture Search (NAS)

Given we've now seen quite some building blocks, and shown that we can calculate their cost and weigh their effectiveness, how do we select such building blocks now? *This is the job of NAS.*

### What is NAS?

We usually design models manually, both the components and their specifics, e.g. channel count, number of layers, resolution, etc. This is black magic. *Getting the model to exactly fit the demand of hardware (compute and memory alike) is super hard.*

*We can use machine learning to design the architectures for us.* On the left, you can see manually designed models. Ideally, a model should be low on MACs, and high on accuracy. *On the right, models developed via NAS are added,* approaching this efficiency requirement much more closely:<br>
<img src="./attachments/Pasted%20image%2020260705170806.png" width="850" /><br>
*Source:* [\[Cai et al., 2020\]](https://arxiv.org/abs/1908.09791)

>[!NOTE]
> Given the huge design space, manual network design is unscalable.
> Therefore, NAS has the goal of finding the best neural network architecture while maximizing the objective(s) of interest (which may e.g. be accuracy, efficiency, latency, storage etc).
> *This is an iterative process.*

<img src="./attachments/Pasted%20image%2020260705183448.png" width="500" /><br>
*Source:* [\[Elsken et al., 2019\]](https://arxiv.org/abs/1808.05377)

### Search Space

*The search space of candidate architectures to choose from is subdivided into:*
- Cell-level Search Space
- Network-level Search Space

#### Cell-Level Search Space

Cells are reusable building blocks that appear multiple times throughout a network. Their architecture is searched once and then reused at every occurrence.<br>
For example, all `Reduction Cells` in a network may be implemented using e.g. pooling or strided convolutions.<br>
We at once optimize the implementation of every `Reduction Cell` simultaneously.

<img src="./attachments/Pasted%20image%2020260705185826.png" width="550" /><br>
*Source:* [\[Zoph et al., 2018\]](https://arxiv.org/abs/1707.07012)

This makes the architecture search faster and simpler, through more coarse.
**What's the architecture to select such choices?**

*We could for example use an RNN to generate candidate cells by:*
- Selecting two previous hidden states from the current pool of available states
  (which in this paper was the prior hidden state and the one before that)
- Selecting input transformation operations (e.g. convolution, pooling, identity), one per hidden state
- Selecting the operation by which to combine the resulting hidden state
- Repeat this process $B$ many times

<img src="./attachments/Pasted%20image%2020260705213720.png" /><br>
*Source:* [\[Zoph et al., 2018\]](https://arxiv.org/abs/1707.07012)

*Per cell:*
- For each hidden state, we can either choose the previous layer or the one before that
- We have $M$ possible operations to perform per each hidden state
- We have $N$ possible operations to combine the resulting hidden state

Hence the design space covers $(2\times 2\times M\times M\times N)^B = 4^BM^{2B}N^B$. With just $M=5,\ N=2,\ B=5$, we would already have to cover $3.2\times 10^{11}$ candidate configurations for cells.

Ultimately, this approach builds cells like this:<br>
<img src="./attachments/Pasted%20image%2020260705214854.png" width="450" /><br>
*Source:* [\[Zoph et al., 2018\]](https://arxiv.org/abs/1707.07012)

>[!NOTE]
> This approach to designing the network can become problematic.
> We had to presume the size of the network and count and placement of the different blocks.
> Also, demands to layer design change from deeper layers to shallow layers.
> It may not be the best choice to just repeat the same building blocks/cells throughout.

#### Network-Level Search Space

Unlike cell-level search, network-level search treats the *entire network as the design space* rather than assuming a single cell architecture is repeated throughout.
This *allows different stages of the network to specialize according to their role,* instead of enforcing a uniform design across all layers.

*Specifically, viewing the network as a whole, the search process can optimize:*
- Different architectures for different blocks
- The network depth
- The width (number of channels) of each block
- The feature resolution (spatial width and height) processed by each block
- The kernel size used in each block

Besides optimizing the individual blocks in a network's context, *network-level NAS can also search over the network topology itself,* that is, how information flows between blocks and across resolutions.

One way to represent this search space is as a graph whose nodes correspond to feature resolutions and whose edges represent candidate transitions between them:<br>
<img src="./attachments/Pasted%20image%2020260705231017.png" width="600" /><br>
*Source:* [\[Liu et al., 2019\]](https://arxiv.org/abs/1901.02985)

This graph shows the convolution-deconvolution path as it would be for the U-Net architecture in the network-level search space.

*Auto-DeepLab extends this idea by jointly searching both the cell architecture and the network-level topology,* allowing the model to automatically discover effective encoder-decoder structures for semantic segmentation:<br>
<img src="./attachments/Pasted%20image%2020260705231156.png" width="600" /><br>
*Source:* [\[Liu et al., 2019\]](https://arxiv.org/abs/1901.02985)

### Design The Search Space

We've seen earlier that just the Cell-Level Search Space encompasses quite a lot of possible combinations. We just blew that cell-level approach up to the network-level, and added multiple additional perspectives such as the individualization of block configurations and the network topology as a whole. *The search spaces explodes multiple times over.* **How do we narrow it down?**

*TinyNAS* is an interesting idea in that regard, as it (1) conducts an *automated search space optimization* and then (2) Performs the model configuration specialization in such a way that it can happen under notable resource constraints.

<img src="./attachments/Pasted%20image%2020260705234735.png" width="600" />

We want to design for TinyML (not TinyNAS, note). TinyML faces three main constraints: *latency, energy, memory.* Naively, for a given task, we might go and accumulate a set of possible network design choices, train them to the end, and compare their accuracy. *But this is costly. So how can we evalute the networks in a most cost-effective way instead?*

>[!NOTE]
> A great heuristic for accuracy is to check which network configuration is able to achieve more FLOPs under the same memory constraint. **The more compute you can provide to a model under some fixed-size memory, the more capacity you can expose the model to, and the more likely it will then end up yielding a higher accuracy.**

This heuristic was shown to be effective in the work that also provides the below image. We can see that the best configuration is neither of the extremes we'd allow for.
A width of $0.5$ and a resolution of $144$ happens to be best, which is supported by the observed highest FLOPs it achieved:<br>
<img src="./attachments/Pasted%20image%2020260705235931.png" width="700" /><br>
*Source:* [\[Lin et al., 2020\]](https://arxiv.org/abs/2007.10319)

### Search Strategy

"We want to have more FLOPs given the same memory budget" provides us well-needed guidance in the network-level architecture search space. *Now that the direction is clear, how do we do search/explore and then how do we automate that?*

<img src="./attachments/Pasted%20image%2020260706001303.png" width="350" />

#### Grid Search

This is the traditional approach, have some step count and step resolution for parameter values, combine those to a grid, i.e. *the entire design space gets represented as the Cartesian product of single dimension design spaces*

<img src="./attachments/Pasted%20image%2020260706002339.png" width="275" />

Originally, grid search did not apply any heuristic like "more flops means likely more accuracy", meaning we'd have to actually fully train all kinds of networks for different parameter configurations only to compare them by their resulting accuracy in the end. *This was expensive on time and resources.*

We see ourselves confronted with whether to tune one parameter to an extreme or to tune across multiple parameters but only moderately so. *Compound scaling goes for the second option, as this approach shows to produce more robust and efficient networks in the field.*

#### Random Search

Random search samples hyperparameter configurations uniformly at random from the search space, instead of evaluating a structured grid of all combinations.
Despite its simplicity, it is often an effective baseline and is commonly used to verify that the selected hyperparameters (and the optimization pipeline) meaningfully influence model performance.

<img src="./attachments/Pasted%20image%2020260706004808.png" width="700" /><br>
*Source:* [maelfabien.github.io](https://maelfabien.github.io/machinelearning/Explorium4/)

#### Reinforcement Learning

Using reinforcement learning for NAS is conceptually straightforward but often difficult in practice.

An RNN acts as a controller. It sequentially samples an architecture $A$ (e.g., number of filters, kernel sizes, layer choices, etc.).<br>
Each architecture is sampled with probability $p(A)$, which also is defined by the controller.<br>
The sampled architecture is then instantiated and trained (under some fixed compute budget).<br>
After training, it achieves an accuracy $R$, which serves as the reward signal (higher is better).

The controller is then updated using the reward signal from the sampled architecture. Specifically, the probability of generating a sampled architecture $A$, denoted $p(A)$, is adjusted based on its achieved reward $R$.

<img src="./attachments/Pasted%20image%2020260706010057.png" width="725" /><br>
*Source:* [\[Zoph and Le, 2017\]](https://arxiv.org/abs/1611.01578)

#### Gradient Descent (DARTS)

Instead of selecting a single operation (e.g. a $3\times 3$ or $5\times 5$ convolution), DARTS considers multiple candidate operations on each edge at the same time.

<img src="./attachments/Pasted%20image%2020260706012827.png" width="650" /><br>
*Source:* [\[Liu et al., 2019\]](https://arxiv.org/abs/1806.09055)

Each candidate operation has a learnable weight $p_i$. For a given input, every operation is applied to the same input, producing multiple outputs. These outputs are then *combined by a weighted sum using the weights* $p_i$. This produces a single output for that edge.

During training, the entire network (with these weighted edges) is trained normally using gradient descent, and performance is measured on a validation set.
The validation performance provides the signal for how to adjust the operation weights.
Operations that improve validation performance get their weights increased, and vice versa.
After training, the continuous weights are converted into a discrete architecture by selecting the strongest operation (highest weight $p_i$) for each edge.

>[!NOTE]
> This approach's biggest drawback is that all candidate operations must be run to be evaluated at every step.
> Since none are removed during training, this method is compute- and memory-intensive.

It is also possible to *take latency into account for gradient-based NAS* by having it become a term in the loss. We need a latency prediction model for this, e.g. a regressor or lookup table, to not have to run inference per architecture choice and having cost further explode.

<img src="./attachments/Pasted%20image%2020260706013905.png" width="650" /><br>
*Source:* [\[Cai et al., 2019\]](https://arxiv.org/abs/1812.00332)

#### Evolutionary Search

Suppose that our design space is the population, represented here as a *One-For-All Network* from which we can sample distinct architecture choices as *Sub-Networks.* Suppose that our latency and accuracy make for the fitness function.

<img src="./attachments/Pasted%20image%2020260706014906.png" width="750" /><br>
*Source:* [\[Tang et al., 2020\]](https://arxiv.org/abs/2007.16100)

This is no evolution yet. For example, we  still need *mutation*, which takes one network as input.
We can now change the depth of the network, change the size of part-taking stages.
We can also change the operator, i.e. a $3\times 3$ convolution may turn into a $5\times 5$ convolution, at random.

We can also *cross-over, where we take two architectures as input and merge them into a mixed architecture* of components from the two parents, choosing randomly.

<img src="../07%20-%20Neural%20Architecture%20Search%20I/attachments/Pasted%20image%2020261001224627.png" width="750" /><br>
*Source:* [\[Cai et al., 2020\]](https://arxiv.org/abs/1908.09791)

---

<p align="center">
      Notes by <a href="https://github.com/MK2112">MK2112</a>
</p>
