# 04 - Pruning & Sparsity II

- [Quick Recap](#quick-recap)
- [Determining the Pruning Ratio](#determining-the-pruning-ratio)
  - [Layer-wise Sensitivity Analysis](#layer-wise-sensitivity-analysis)
  - [Automatic Pruning](#automatic-pruning)
- [NetAdapt](#netadapt)
- [Fine-Tune / Train Pruned Neural Networks](#fine-tune--train-pruned-neural-networks)
- [System Support for Sparsity](#system-support-for-sparsity)
  - [Efficient Inference Engine: Weight Sparsity + Activation Sparsity](#efficient-inference-engine-weight-sparsity--activation-sparsity)
  - [Sparse Weight](#sparse-weight)
    - [Micro-Architecture of Each Processing Element](#micro-architecture-of-each-processing-element)
  - [NVIDIA Tensor Core: M:N Weight Sparsity](#nvidia-tensor-core-mn-weight-sparsity)
  - [TorchSparse & PointAcc: Activation Sparsity](#torchsparse--pointacc-activation-sparsity)
    - [TorchSparse: Sparse Convolution Library](#torchsparse-sparse-convolution-library)
    - [PointAcc: Hardware Accelerator for Sparse Convolutions](#pointacc-hardware-accelerator-for-sparse-convolutions)

## Quick Recap

Pruning means taking a dense neural network and turning it into a  sparse neural network in such a way that the resulting model saves computation and memory but retains accuracy.
We saw unstructured pruning to allow for the highest pruned weight ratios, while being harder to accelerate than the more structured approaches.

To determine which weights to remove, we may look at their magnitude and then prune smaller weights.<br>
Pruning can also be done on the level of neurons, which is more coarse.

## Determining the Pruning Ratio

We saw in the last lecture that pruning individual channels yields better efficiency, meaning higher accuracy at lower latency, than imposing some threshold across all channels.<br>
*How can we find the pruning ratio for each layer, i.e. how can we find which layer to prune more and which less so as to retain accuracy?*

### Layer-wise Sensitivity Analysis

The pruning criterion can be approximately determined with what's called **sensitivity analysis.**<br>
This is based on the idea that *certain layers to be more sensitive to pruning efforts* than others.

**Given a network, we can brute-force the investigation into which layers are best to prune:**
- Pick some (not yet checked) layer $L_i$ from the model
- Prune $L_i$ with pruning ratios in some strides, e.g. $r \in \{0,\ 0.1,\ 0.2, \dots,\ 0.9\}$, in our case using fine-grained / unstructured pruning
- Observe accuracy degradation $\Delta \text{Acc}_r^i$ for these pruning ratios
- Repeat for all unchecked layers $L$

<img src="./attachments/Pasted%20image%2020260507200531.png" width="475" />

These sensitivity graphs and interpretations *assume independence of the layers, i.e., no co-sensitivities between different layers are investigated.*
This, of course, is a simplified notion.

>[!NOTE]
> Depending on the layer's retained accuracy, we can now set its pruning ratio higher or lower, using the above graphs as guidance.
> Note, again, that this approach is a heuristic at best, as it ignores potential layer co-sensitivities, but affords us insight with relatively few experiments.

### Automatic Pruning

Now, to what extent can we go beyond layer-wise sensitivity analysis in terms of quality of insight?<br>
How can we also address layer co-sensitivities?

We need to *automate, and ideally in a way that finds solutions that are not only good, but also suitable to our hardware.*<br>
[AutoML](https://arxiv.org/abs/1802.03494) figures out the pruning ratio and the pruning strategy for different neural networks by *utilizing an actor-critic agent structure.*

**We can express the AutoML Model Compression setup in the terms of reinforcement learning:**
- *State:* Features / Embeddings including layer indices, channel numbers, kernel sizes, FLOPs, ...
- *Action:* The actor imposes some pruning ratio as a continuous number $a \in [0,1)$
- *Agent:* A deep deterministic policy gradient-based agent, supporting continuous action output
- *Reward:*

$$
R = \begin{cases}
-\text{Error} \cdot \log(\text{FLOP}), & \text{if constraints are satisfied} \\
-\infty, & \text{otherwise}
\end{cases}
$$

**If accuracy gets better and FLOPs decrease, reward grows.**

The image shows the reward as $\text{Reward} = -\text{Error}$, but usually it is $\text{Reward} = -\text{Error} \cdot \log(\text{FLOP})$ to both *encourage retaining accuracy and processing efficiency:*

<img src="./attachments/Pasted%20image%2020260507204343.png" width="475" /><br>
*Source:* [\[He et al., 2018\]](https://arxiv.org/abs/1802.03494)

AutoML shows to be quite effective, beating human skill in terms of archieved pruning ratio for ResNet50 and taking less time to achieve this:<br>
<img src="./attachments/Pasted%20image%2020260507210404.png" width="600" /><br>
*Source:* [Efficient Methods and Hardware for Deep Learning \[Han S., Thesis, Stanford University\]](https://stacks.stanford.edu/file/druid:qf934gh3708/EFFICIENT%20METHODS%20AND%20HARDWARE%20FOR%20DEEP%20LEARNING-augmented.pdf)

Zooming into this work, done on ResNet-50, we find that AutoML correctly identifies that $1 \times 1$ / point-wise convolutional layers impose less redundancy, therefore showing themselves to *not allow* for high pruning ratios. The $3\times 3$ convolutional layers show more such redundancy and thus afford higher pruning ratios.

<img src="./attachments/Pasted%20image%2020260507210913.png" width="700" /><br>
*Source:* [\[He et al., 2018\]](https://arxiv.org/abs/1802.03494)

>[!CAUTION]
>**Question:** Why are $1\times 1$ convolutions less prunable than $3\times 3$ convolutions in ResNet-50?

>[!TIP]
>**Answer:** The $3 \times 3$ convolutions contain more weights interacting with each channel, which potentially can cause overlapping or correlated spatial filters to be learnt. This 'higher bandwidth into each channel' affords higher ratios of pruning without having communication from said channels collapse, as it would be the case with the  $1\times 1$ convolutions, which communicate in a much more "narrow bandwidth" with each input channel (1 learnable parameter/input channel). Removing these parameters therefore has a much higher, potentially detrimental effect on information flow. 

*Different objectives will lead AMC to provide differently pruned results.*<br>
For example, objectives may be saving FLOPs, saving time, or both, as mentioned before:

<img src="./attachments/Pasted%20image%2020260507212734.png" width="700" /><br>
(The $0.75$ MobileNet is a uniformally pruned version, pruning every layer by the same ratio)

## NetAdapt

Conceptually similar to AutoML, NetAdapt's goal now is to *find a per-layer pruning ratio to meet a global resource constraint* (e.g., latency, energy). For example, suppose that for each iteration, we aim to reduce the latency by a certain reduction amount $\Delta R$ (some manually defined value).

*Per iteration:*
- We tune one layer $L_k$ ($k$ is $\text{A}\dots \text{Z}$ in the figure below) with the intent of it producing the $\Delta R$ latency reduction
- After we hit the $\Delta R$ target for the layer, we fine-tune the model ($\approx 10k$ iterations, not much more)
- We then measure the accuracy $Acc_k$ for the fine-tuned $\Delta R$-hitting model
- This is repeated for every layer in $L$

>[!NOTE]
> Ultimately, we choose the configuration for some pruned layer $k$ where the resulting accuracy degradation is the smallest.
> That single adapted layer $k$ is the change we then implement for reaching $\Delta R$ latency reduction with our final model.
> We then run this in an outer loop until we reach some target latency reduction $n \cdot \Delta R$ for the model as a whole, by adding up the selected layer-wise partial latency improvements
> (that cost the least accuracy individually).

<img src="./attachments/Pasted%20image%2020260507215127.png" width="700" /><br>
*Source:* [\[Yang et al., 2018\]](https://arxiv.org/abs/1804.03230)

*Note three important things on NetAdapt:*
- This idea still somewhat sees layers as independent units to be optimized, incorporating regard for layer co-sensitivities only through a selection by accuracy
- We can, for the final model configuration, afford to additionally fine-tune for longer, helping in potentially recovering yet more final model accuracy
- We obtain a serial of models with different costs as a byproduct

<img src="./attachments/Pasted%20image%2020260922224249.png" width="650" /><br>
*Source:* [\[Yang et al., 2018\]](https://arxiv.org/abs/1804.03230)

## Fine-Tune / Train Pruned Neural Networks

We already discussed fine-tuning as a way to recover much of the original model's accuracy after pruning a trained full-size network. In that fine-tuning, the learning rate is $\sim 1/100\times\dots1/10\times$ the original learning rate so as to not uproot the weights of an already converged model, but to just lightly adapt them.

Yet another improvement comes from iterative pruning and fine-tuning:
<img src="./attachments/Pasted%20image%2020260503154349.png" width="650" /><br>
*Source:* [\[Han et al., 2015\]](https://arxiv.org/abs/1506.02626)

Just like during initial training, the *network weights have to be regularized* during fine-tuning so as to penalize non-zero weights and thus encourage the effective network size (non-zero weight count) to generally become more contained and thus more prunable.

Sometimes $L1$-Regularization works better for this, sometimes $L2$-Regularization:
- $L1$-Regularization:   $L' = L(\textbf{x};\ \textbf{W}) + \lambda |\textbf{W}|$
- $L2$-Regularization:   $L' = L(\textbf{x};\ \textbf{W}) + \lambda ||W||_2$

>[!CAUTION]
>**Question:** I do not see the connection. How does regularization support us in producing a more easily prunable network?

>[!TIP]
> **Answer:** Pruning aims to remove weights that are small enough to not contribute or alter information flow meaningfully in the network. Regularization creates more such small weights, i.e. pruning candidates.<br>
> Usually we would finetune to only minimize $L(\textbf{x};\ \textbf{W})$ we now need to minimize $L(\textbf{x};\ \textbf{W}) + \lambda R(\textbf{W})$ with either $R(\textbf{W}) = |\textbf{W}|$ (L1, the absolute sum of the values in $\textbf{W}$) or $R(\textbf{W}) = ||\textbf{W}||_2$ (L2, the root of the sum of the squared values of $\textbf{W}$).<br>
> In this sense, as $R(\textbf{W})$ now weighs into the loss we optimize for, we also minimize $R(\textbf{W})$, meaning the magnitudes of the values making up $\textbf{W}$. It is this dynamic that we utilize to better find prunable weights.

## System Support for Sparsity

Once we have the pruned model, obtained through e.g. repeated pruning and fine-tuning, how can we then best leverage the budget that becomes available from said pruning?

### Efficient Inference Engine: Weight Sparsity + Activation Sparsity

The idea remains to apply model compression (pruning and quantization), but we then *provide measures for inference accelearation on the system-side* on top of that.

**An efficient inference engine to accelerate an already compressed network can take advantage of three aspects:**
- *Weight Sparsity:* Any zero-weight multiplied by some activation returns zero. We could outright skip calculating these, reducing computation and memory demands (memory savings are lower because we have to store indices for where such skippable weights are). This is static, applied like a mask.
- *Activation Sparsity:* No matter how big the weight, if an activation happens to be zero, the product will be zero. This doesn't alter the weight memory, only saving on computation. This is dynamic, input-dependent and thus harder to realize.
- *Weight Sharing:* Reducing the numerical resolution allows for re-using weights at different positions. This reduces only the memory footprint. *We will discuss this further in the next lecture.*

<img src="./attachments/Pasted%20image%2020260512180702.png" width="650" /><br>
*Source:* [\[Han et al., 2016\]](https://arxiv.org/abs/1602.01528)

### Sparse Weight

To effectively parallelize the computation on *sparse weight matrices,* suppose this example of a $1 \times 4$ vector being multiplied with a sparse $8\times 4$ matrix, which then gets ReLU-activated.

For this sparse matrix, we *only store the non-zero* weights. *We have four processing elements* $PE0$ to $PE3$, which can process their allocated data in parallel.

<img src="./attachments/Pasted%20image%2020260512203657.png" width="400" /><br>
*Source:* [\[Han et al., 2016\]](https://arxiv.org/abs/1602.01528)

The below table shows how $PE0$ stores its assigned weights (green in the above matrix).<br>
<img src="./attachments/Pasted%20image%2020260512203717.png" width="400" /><br>
*Source:* [\[Han et al., 2016\]](https://arxiv.org/abs/1602.01528)

From the above matrix, *$PE0$ is responsible for processing the green rows.*
Listing only the non-zero values from these rows in *column-major order* yields:
$$[w_{0,0},\ w_{0,1},\ w_{4,2},\ w_{0,3},\ w_{4,3}]$$
Now, the *relative index* expresses how many zeros were skipped since the previous non-zero value, column-wise. For example, $w_{0,0}$ appears immediately, but then is followed by a zero entry in its column. The next column begins with $w_{0,1}$, but since the last column contained a zero, the relative index of $w_{0,1}$ is $1$. $w_{0,1}$ is followed by $2$ zeros until we encounter $w_{4,2}$, and then immediately we encounter $w_{0,3}$ and $w_{4,3}$, yielding these relative indices:
$$[0,1,2,0,0]$$
The *column pointer*, following the standard *CSC format*, expresses where each column's data begins when only looking at the *column-major ordered* non-zero weight vector $[w_{0,0},\ w_{0,1},\ w_{4,2},\ w_{0,3},\ w_{4,3}]$:
- Column $0$ of the matrix contains $1$ weight value, $w_{0,0}$
	- Column $0$ starts with the $0^{\text{th}}$ entry of the non-zero weight vector
- Column $1$ of the matrix contains $1$ weight value, $w_{0,1}$
	- Column $1$ starts with the $1^{\text{st}}$ entry of the non-zero weight vector
- Column $2$ of the matrix contains $1$ weight value, $w_{4,2}$
	- Column $2$ starts with the $2^{\text{nd}}$ entry of the non-zero weight vector
- Column $3$ of the matrix contains $2$ weight values, $w_{0,3}$ and $w_{4,3}$
	- Column $3$ starts with the $3^{\text{rd}}$ entry of the non-zero weight vector
	- Column $3$ occupies indices $3$ and $4$ in the non-zero weight vector
	- The $4^{\text{th}}$ index is not beginning a new column, therefore it cannot
	  just be noted as column pointer $4$
	- Instead, the next free position after column $3$ is at index $5$, and CSC stores this end position as the final column pointer, expressing that there must be two values in the last column

*Hence the column pointers become:*
$$[0,1,2,3,5]$$

>[!NOTE]
> It is these column pointers and relative indices that cause the memory optimization to not be as good as the computational optimization for sparse weights.

In practice, we can skip those calculations associated with an input activation $0$. We only need to broadcast non-zero activations to non-zero weight entries now, effectively skipping zero computations.

<img src="./attachments/Pasted%20image%2020260512221339.png" width="400" /><br>
*Source:* [\[Han et al., 2016\]](https://arxiv.org/abs/1602.01528)

#### Micro-Architecture of each Processing Element

At this point, we reduced the amount of computation for each processing element $PE$, but *what does such a $PE$ do* in the first place?

Each $PE$ receives non-zero activations together with their activation indices (where in the activation vector they are). These are stored inside the _Activation Queue_.

Each activation index is used to access the corresponding column of the compressed sparse weight matrix. Using the *column pointers* (remember, these are part of the presentation of the sparse matrix in SRAM now), the $PE$ locates where the non-zero weights for that column begin and end inside the *Sparse Matrix in the SRAM*.

The *Weight Decoder* and *Address Accumulator* reconstruct the destination row addresses from the relative indices that also are part of the *Sparse Matrix in SRAM*.

Each decoded weight then gets multiplied with its column's activation value. The result is accumulated into the corresponding output activation register for the decoded destination row. The ReLU unit applies the activation function to said accumulated values. Finally, the *Leading Non-Zero Detect* unit keeps only non-zero output activations to be forwarded for the next layer.

<img src="./attachments/Pasted%20image%2020260512224132.png" width="800" /><br>
*Source:* [\[Han et al., 2016\]](https://arxiv.org/abs/1602.01528)

>[!NOTE]
> We have this *Leading NZero Detect* take advantage of the activation sparsity to reduce computation for the next layer. Equally, we take advantage of weight sparsity by storing and processing only non-zero values from the *Sparse Matrix SRAM* (i.e. the sparse weight matrix).
> **Remember, this setup is not intended for training, but for fast, efficient inference.**
> Also note that we use SRAM, not DRAM.

This idea of focusing operations to non-zero weight and activation entries has substantial impact on various different network types, providing at least an order of magnitude improvement:<br>
<img src="./attachments/Pasted%20image%2020260512230130.png" width="500" /><br>
*Source:* [\[Han et al., 2016\]](https://arxiv.org/abs/1602.01528)

| EIE Main Pro                                                                                                                                                                                                       | EIE Main Con                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Demonstrated that special-purpose hardware can make it cost-effective to do up to 50% sparse matrix operations, while exploiting both weight sparsity and activation sparsity, saving both compute time and energy | Fits everything into SRAM; practical for TinyML, not for LLMs, though this is becoming more and more popular for decoding |

>[!NOTE]
>The first principle of efficient AI computing is to be lazy: avoid redundant computation, quickly reject the work, or delay the work. We envision future AI models will be sparse at various granularity and structures. Co-designed with specialized accelerators, sparse models will become more efficient and accessible.<br>
>*Source:* [\[Han, et al. 2023\]](https://arxiv.org/abs/2306.09552)

### NVIDIA Tensor Core: M:N Weight Sparsity

In the last lecture, we already saw the below $2:4$ sparsity configuration for a weight matrix.
A standard case for this is 2:4 sparsity (50% pruning):<br>
<img src="./attachments/Pasted%20image%2020260503183906.png" width="400" /><br>
*Source:* [\[Mishra et al., 2021\]](https://arxiv.org/abs/2104.08378)

The $2:4$ sparse matrix then, especially on NVIDIA Ampere GPUs can be very effectively subdivided and reduced to two dense matrices holding all the information necessary to reconstruct the sparse original:<br>
<img src="./attachments/Pasted%20image%2020260512233626.png" width="400" /><br>
*Source:* [\[Mishra et al., 2021\]](https://arxiv.org/abs/2104.08378)

In particular, per row, we can represent the information of $8$ elements through $4$ non-zero elements. But, as already mentioned, *we also need the matrix of 2-bit indices per non-zero weight to represent the location of said non-zero weight.* For the non-zero weight, the group is already indicated by its position in the compressed matrix. Still, the non-zero weight may be in one of four possible locations within its group.

Because of the $2$-bit indices masking out the non-zero matrix contents, they effectively reduce the FLOPs.<br>
In the $2:4$ sparsity configuration, shown on the right in the below image, this amounts to only $2$ multiplications out of $4$ being *actually* calculated ($50\%$ FLOPs reduction).

<img src="./attachments/Pasted%20image%2020260514230247.png" width="800" /><br>
*Source:* [\[Mishra et al., 2021\]](https://arxiv.org/abs/2104.08378)

Note that the selector process requires some (very minimal) overhead.<br>
Still, the speedup effects increase with the size of the dense matrix that we apply $2:4$ sparsity to. They nearly approach the $2\times$ improvement level. At the same time, accuracy remains pretty much unaffected.

<img src="./attachments/Pasted%20image%2020260514231939.png" width="650" /><br>
*Source:* [\[Mishra et al., 2021\]](https://arxiv.org/abs/2104.08378)

### TorchSparse & PointAcc: Activation Sparsity

Coming from the discussion of weight sparsity, TorchSparse now serves as an example supporting Activation Sparsity.
#### TorchSparse: Sparse Convolution Library

To understand what sparse convolutions are, suppose a sparse input feature map, i.e. a matrix where most entries are $0$.  In a standard $3 \times 3$ convolution, every output pixel is a result of an "informed look around its neighborhood" via the kernel weights. Equally, a non-zero input value affects every output whose receptive field include it.

After a few convolutional layers, the single non-zero input value spreads and dilates to surrounding locations, having them become non-zero. *The feature map therefore, through these few convolutional layers, becomes dense. Where there was sparse input data, the sparsity now becomes lost.*

The sparse convolution works against this dilation effect. Instead of producing outputs everywhere the kernel touches, outputs are only computed at locations in the output that were already non-zero in the input. *This preserves the distinct non-zero coordinates.*

If our input feature map contains only $\sim 0.01\%$ non-zero datapoints, the positions (and therefore the ratio) of the non-zeros will be maintained through the special case *submanifold sparse convolution.*

<img src="./attachments/Pasted%20image%2020260515000620.png" width="350" /><br>
*Source:* [\[Graham, 2015\]](https://arxiv.org/abs/1706.01307)

*Submanifold sparse convolution affords to skip calculations the kernel would produce for output locations unequal to the non-zero's input location.* Note that the below image calls those calculations 'matrix multiplications' because the visualization omits the channel dimension.

<img src="./attachments/Pasted%20image%2020260515003914.png" width="600" /><br>
*Source:* [\[Tang et al., 2022\]](https://arxiv.org/abs/2204.10319)

>[!NOTE]
> The feature map governs the relationship between the input activation, the output activation and the weight.

*Effectively, sparse convolution is implemented as a list of the valid input-output interactions.* For example, $(P_1,Q_3,W_{-1,0})$ means that the non-zero input feature $P_1$ contributes to the non-zero output feature $Q_3$, where this specific contribution is shaped through the kernel weight at offset $(-1,0)$.
Note that the convolution operation we reference here states the kernel to look like this:

$$W=
\begin{bmatrix}
W_{-1,-1} & W_{-1,0} & W_{-1,1} \\
W_{0,-1} & W_{0,0} & W_{0,1} \\
W_{1,-1} & W_{1,0} & W_{1,1}
\end{bmatrix}$$
For the actual, applied convolution operation, this gets flipped horizontally and vertically:
$$\begin{bmatrix}
W_{1,1} & W_{1,0} & W_{1,-1} \\
W_{0,1} & W_{0,0} & W_{0,-1} \\
W_{-1,1} & W_{-1,0} & W_{-1,-1}
\end{bmatrix}$$

<img src="./attachments/Pasted%20image%2020260515020445.png" width="600" /><br>
*Source:* [\[Tang et al., 2022\]](https://arxiv.org/abs/2204.10319)

>[!NOTE]
> It is a property of this submanifold sparse convolution approach that *the kernel weight $W_{0,0}$ appears in the computation of every non-zero spatial location as a self-connection. This is reflected in the list titled $\text{Maps}$ in the above image.

>[!CAUTION]
> **Question:** I do not understand how the kernel's $W_{0,0}$ can contribute to every non-zero input-output pair found at the same spatial coordinate, especially for $P_3$ and $Q_3$. Wouldn't that require padding? Am I misunderstanding something?

>[!TIP]
> **Answer:** This is a common misunderstanding. The $\text{Maps}$ list can mislead if interpreted using the standard, dense convolution intuition. In submanifold sparse convolution, *we do not follow the established image of sliding a kernel over a (full) grid.* Instead, a computation graph is defined over *only the non-zero spatial locations.* The kernel offset $(0,0)$ then corresponds to a weighted self-connection and every non-zero input location $P_i$ always contributes to its output $Q_i$ at that same location.

Depending on how the non-zero inputs are distributed across the feature map, different kernel weights may participate in different numbers of valid input-output interactions, see the amount of interactions for $P_2$ vs. $P_3$ above, respectively. As a result, the workloads per kernel weight may differ from one another, causing an operational imbalance and inefficiency.

*This is (at least partially) addressed through adaptive grouping.* This is a technique which groups similarly sized matrix multiplications from different kernel weights together and executes them as batches. This 'load balancing' improves the regularity and utilization of GPU computation, but it also introduces some overhead for organizing and padding the groups accordingly.

<img src="./attachments/Pasted%20image%2020260515024029.png" width="600" /><br>
*Source:* [\[Tang et al., 2022\]](https://arxiv.org/abs/2204.10319)

TorchSparse++ is an increment that should be mentioned. It implements improved memory and computation overlap methods. One such method consists of re-ordering data in memory.

Suppose we use *vanilla TorchSparse.* Further suppose we have our *$3 \times 3$ kernel weights* (black tiles below). Now, for example, the output $B_0$ (first red tile) is produced through summing the multiplications of input activation $A_0$ (first grey tile) with $W_{0,0}$, $A_1$ with $W_{0,1}$, and $A_3$ with $W_{1,1}$.
As the kernel slides over the sparse input feature map, like before, different weights may react with different inputs, or may not react at all.

<img src="./attachments/Pasted%20image%2020260519181529.png" width="300" /><br>
*Source:* [\[Tang et. al, 2023\]](https://openaccess.thecvf.com/content/CVPR2023W/WAD/papers/Tang_TorchSparse_Efficient_Point_Cloud_Engine_CVPRW_2023_paper.pdf)

As already mentioned, the *adaptive grouping* gets applied to the input activations for batched output generation. In this example, the batch size is $2$.

For example, the calculations for $B_0$ and $B_1$ are batch-processed. Within this batch, $A_0$ needs to be multiplied with $W_{0,-1}$ for $B_1$, while no such calculation with $W_{0,-1}$ is necessary for $B_0$. We have to include placeholder operations for $B_0$ (see the first two light red tiles). This is an excess we have to account for so as to maintain batch structures, amounting to $12$ such redundant computations in total.

The first plus in TorchSparse++ refers to *Row Reordering.* We re-organize batches by aligning rows (i.e. the calculations of distinct outputs) according to their similarity in order to reduce the redundant computation.

<img src="./attachments/Pasted%20image%2020260519202852.png" width="300" /><br>
*Source:* [\[Tang et. al, 2023\]](https://openaccess.thecvf.com/content/CVPR2023W/WAD/papers/Tang_TorchSparse_Efficient_Point_Cloud_Engine_CVPRW_2023_paper.pdf)

The second plus refers to *Column Splitting.* The idea is to split the calculation of outputs $B_i$ across three partially contributing groups. Crucially, within each group, the target outputs can be arranged independently according to the above *row reordering* principles. Instead of memory contributing to the computation of only a single output at a time, like we saw before, different memory regions now calculate partial results for different outputs simultaneously.

Because these groups are smaller, row reordering becomes more effective. Redundant computations are confined to individual batch groups, reducing their overall scope.

<img src="./attachments/Pasted%20image%2020260519205126.png" width="300" /><br>
*Source:* [\[Tang et. al, 2023\]](https://openaccess.thecvf.com/content/CVPR2023W/WAD/papers/Tang_TorchSparse_Efficient_Point_Cloud_Engine_CVPRW_2023_paper.pdf)

>[!NOTE]
> TorchSparse++ requires a little bit of organizational processing overhead for grouping and re-ordering and ultimately re-assembling the results. The achievable gains scale and thus will outpace the cost of this overhead, though.

#### PointAcc: Hardware Accelerator for Sparse Convolutions

As we saw, TorchSparse++ optimizes how sparse convolution computations are scheduled and reordered to reduce redundant computation and improve memory efficiency.

PointAcc now targets a broader bottleneck occuring earlier during processing. It *addresses how to determine non-zero outputs and input-output relationships efficiently* along with reducing data movement.

Suppose we apply PointAcc to the kernel weight $W_{-1,-1}$. With stride $1$, our sparse convolution kernel would align the input and output points exactly on top of one another. 

Now, since convolution flips the kernel horizontally and vertically, PointAcc says that $W_{-1,-1}$ corresponds to shifting the input one position right and one position down from the kernel center.
We therefore *shift the input by $(1,1)$* like so:

<img src="./attachments/Pasted%20image%2020260520005419.png" width="350" /><br>
*Source:* [\[Lin et al., 2021\]](https://dspace.mit.edu/bitstreams/f3a5c501-09b3-40a2-b564-57da7df1d3c7/download)

PointAcc now compares the positions of the input and output values in memory, represented here as coordinates. While the input coordinates are shifted by $(1,1)$, the output coordinates remain unchanged. *A merge sort is then applied to combine the shifted input and original output coordinate sets.* In the resulting list, *matching neighboring coordinates indicate an intersection.* Specifically the finding of these intersections between the two merged arrays can be accelerated by hardware.

<img src="./attachments/Pasted%20image%2020260520010832.png" width="350" /><br>
*Source:* [\[Lin et al., 2021\]](https://dspace.mit.edu/bitstreams/f3a5c501-09b3-40a2-b564-57da7df1d3c7/download)

>[!NOTE]
> Remember, this isn't standard convolution. The kernel behaves differently in that our offset $(-1,-1)$ corresponds to a weighted connection of any non-zero input location $P_{ij}$ to its output $Q_{i+1,j+1}$.

We can now repeat this for all the kernel weights. In that sense, similar for $W_{1,1}$, we'd shift by $(-1,1)$ (i.e. into the top left corner). Overall PointAcc methods provide both multiples in improvement in speedup and energy saving.

<img src="./attachments/Pasted%20image%2020260520011912.png" width="350" /><br>
*Source:* [\[Lin et al., 2021\]](https://dspace.mit.edu/bitstreams/f3a5c501-09b3-40a2-b564-57da7df1d3c7/download)

---

<p align="center">
      Notes by <a href="https://github.com/MK2112">MK2112</a>
</p>
