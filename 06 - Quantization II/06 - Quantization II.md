# 06 - Quantization II

- [Quick Recap](#quick-recap)
- [Post-Training Quantization (PTQ)](#post-training-quantization-ptq)
  - [Quantization Granularity](#quantization-granularity)
    - [Per Tensor Quantization](#per-tensor-quantization)
    - [Per-Channel Quantization](#per-channel-quantization)
    - [VS-Quant](#vs-quant)
    - [Group Quantization](#group-quantization)
  - [Dynamic Range Clipping](#dynamic-range-clipping)
  - [Rounding](#rounding)
- [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
- [Binary and Ternary Quantization](#binary-and-ternary-quantization)
  - [Binary Weight Network (BWN)](#binary-weight-network-bwn)
  - [Binary Quantizing Both Weights and Activations](#binary-quantizing-both-weights-and-activations)
  - [Ternary Weight Networks (TWN)](#ternary-weight-networks-twn)
  - [Trained Ternary Quantization (TTQ)](#trained-ternary-quantization-ttq)
- [Automatic Mixed-Precision Quantization](#automatic-mixed-precision-quantization)

## Quick Recap

We left off with an understanding of these three levels of quantization:<br>
<img src="./attachments/Pasted%20image%2020260627234404.png" width="550" />

- We first looked at how to *efficiently allocate memory for floating point values*
- Then we applied *K-Means-based Quantization, which saves storage but not compute*
	- This is still quite useful for LLM inference if batch sizes are small; we are rather memory-bounded than compute-bounded here, so we can afford floating-point arithmetic
- Applying Linear Quantization allows for both the weights and the activations to be represented through integer arithmetic
	- The *Scaling Factor* $S$ maps the integer range of $q$ to the floating-point range of $r$
	- The *Zero Point* $Z$ maps a point in the integer range of $q$ to be zero in the floating-point range of $r$

Best efficiency improvements can be obtained by *combining (in order) pruning and quantization:*<br>
<img src="./attachments/Pasted%20image%2020260627234404.png" width="650" /><br>
*Source:* [\[Han, et al. 2016\]](https://arxiv.org/abs/1510.00149)

## Post-Training Quantization (PTQ)

We saw thesee parameters already, but now, how can we actually obtain optimal linear quantization parameters $S$ and $Z$?

### Quantization Granularity

Different qualities emerge depending on the degree to which we apply quantization across the tesnor. *Three distinct main granularities for quantization have emerged:*<br>
<img src="./attachments/Pasted%20image%2020261029222154.png" width="400" /><br>

#### Per Tensor Quantization

With *Per Tensor Quantization*, we use *one scaling factor to scale the entire tensor.*
This approach works quite well for large models, but *causes accuracy drops in smaller models.*

![](./attachments/Pasted%20image%2020260628140553.png)
*Source:* [\[Nagel et al., 2019\]](https://arxiv.org/abs/1906.04721)

>[!Question] Why is it that Per Tensor Quantization works poorly for smaller models?
> The value ranges differ noticably ($\gt 100 \times$) per channel. *Using a single scaling factor across such diverse ranges causes channels with smaller values to lose precision due to coarse quantization needed for the larger ranges.* This loss of precision occurs in both small and large models, but *smaller models have less redundancy and representational capacity to compensate* for the resulting errors.

Looking at an example for per tensor quantization, suppose we want to quantize this weight matrix with *symmetric 2-bit linear quantization:*
![169](./attachments/Pasted%20image%2020260628223951.png)
($ic$ means input channels, $oc$ means output channels)

1. We first identify $|r|_{max} = 2.12$ from the weight matrix above
2. We then calculate for $N=2\ \text{Bit}$ our symmetric scaler $S = \frac{|r|_{max}}{q_{max}} = \frac{2.12}{2^{2-1}-1} = 2.12$
3. We know $Z = 0$ for this *symmetric setting,* so we calculate $q = \text{round} \left( \frac{r}{S} \right) = \text{round} \left( \frac{2.09}{2.12} \right) = 1$

Repeated for all weights, we obtain the tensor quantized weight matrix:
![143](./attachments/Pasted%20image%2020260628153719.png)

Reconstruction via $r = (q - Z) \times S = (q - 0) \times 2.12$ yields:
![313](./attachments/Pasted%20image%2020260929222052.png)

Luckily, we can avoid per tensor quantization's loss of well needed resolution for channels with smaller value ranges by quantizing *not across, but per each channel* instead. The difference amounts to $||W - Sq||_F = 2.28$.

#### Per-Channel Quantization

WIth *Per-Channel Quantization*, each *output channel* of the tensor is scaled by one scaling factor

Let's stick with the same set of weights to quantize:
![202](./attachments/Pasted%20image%2020260628223916.png)

1. For the first row, we find $|r|_{max} = 2.09$ and $S_0 = \frac{|r|_{max}}{q_{max}} = \frac{2.09}{2^{2-1}-1} = 2.09$
2. For the second row, we find $|r|_{max} = 2.12$ and $S_1 = \frac{|r|_{max}}{q_{max}} = \frac{2.12}{2^{2-1}-1} = 2.12$
3. For the third row, we find $|r|_{max} = 1.92$ and $S_2 = \frac{|r|_{max}}{q_{max}} = \frac{1.92}{2^{2-1}-1} = 1.92$
4. For the fourth row, we find $|r|_{max} = 1.87$ and $S_3 = \frac{|r|_{max}}{q_{max}} = \frac{1.87}{2^{2-1}-1} = 1.87$
5. We know $Z = 0$ for each row, as they are symmetrically quantizable: Per row, we calculate the values via $q_{n,m} = \text{round} \left( \frac{r_m}{S_n} \right)$

*Applying quantization channel-wise yields a smaller reconstruction error,* as we can much better find scaling factors now, maintaining more resolution across the overall weight matrix:
![272](./attachments/Pasted%20image%2020260628164600.png)

The problem with *per channel quantization* is that it flat-out subdivides row-wise, not by any measure of effective grouping. Also, *the multiple quantization scales mean more storage demand than required for per tensor quantization.*

> If you favor your model to be parallelizable, go for coarse quantization (like whole-tensor quantization). If you want to focus on retaining as much accuracy as possible, go for finer quantization (like per-channel, but its not good, rather see next few methods)

#### VS-Quant

VS-Quant is an approach that can get us closer to achieve a real *balance between quantization accuracy and hardware efficiency.* 

![435](./attachments/Pasted%20image%2020260628211149.png)
*Source:* [\[Dai et al., 2021\]](https://arxiv.org/abs/2102.04503)

**We now have a two-layered approach:** 
$$r = S \cdot (q-Z) \rightarrow r = \gamma \cdot S_q \cdot \left( q-Z \right)$$
- Each row now gets subdivided into vectors
- Each vector $q$ now *locally* gets scaled by its own *integer per-vector scale factor $S_q$*
- The tensor as a whole, *globally,* is scaled yet again by $\gamma$, a *floating-point coarse grained scaling factor*

The per-tensor (global) scaling factor $\gamma$ is the more expensive one, but its coarse grained (affects / gets affected by the entire tensor), as opposed to the fine-grained (only regard their vector $q$) per-vector scale factors $S_q$.

For the above shown example, suppose a group is defined to hold $16$ elements, we want to achieve $4$-bit-quantization and a $4\ \text{Bit}$ per-vector scaling factor, we can calculate the bit width of each weight:
- Each quantized value in the weight matrix occupies $4\ \text{Bit}$
- For every $16$ elements, we also occupy $4\ \text{Bit}$ for the per-vector scaling factor
- The per-vector scaling factor gets amorized across the $16$ vector elements

This means the effective bit width of each value of the weight matrix is:
$$4 + 4 / 16 = 4.25\ \text{Bit}$$
#### Group Quantization

 With *Group Quantization,* we generalize the idea of VS-Quant's two-level quantization approach to allow for any desired amount of levels for quantization.
$$r = S \cdot (q-Z) \rightarrow r = (q-Z) \cdot S_{l_0} \cdot S_{l_1} \cdot \dots$$
**How does this fit into our so-far discussed set of quantization schemes?**

Going back, intuitively, we can visualize [Per-Channel Quantization](#Per-Channel%20Quantization) like so:
![](./attachments/Pasted%20image%2020260628224752.png)

*(Note that before we had drawn channels horizontally from the weight matrix example. In our understanding and context, a "channel" is not inherently horizontal or vertical, but simply one slice of a tensor)*

- A weight ends up represented via `INT4` datatype
- The group across which lowest level scaling `L0` via $S_{l_0}$ gets applied to is the channel as a whole
- The $S_{l_0}$ is of floating-point type `FP16`
- No higher level scaling `L1` gets applied
- *If we assume many thousands of dimensions in the channel, the effective bit width tends towards $4\ \text{Bit}$, easily amortizing the single `FP16` scaling factor for all of the `INT4` values*

In that same context, we can now visualize [VS-Quant](#VS-Quant) like so:
![](./attachments/Pasted%20image%2020260628225955.png)
*Source:* [\[Rouhani et al., 2023\]](https://arxiv.org/abs/2302.08007)

- A weight ends up represented via `INT4` datatype, still
- The group across which lowest level scaling `L0` via $S_{l_0}$ gets applied to is *a subset of the entries in the channel*
- The $S_{l_0}$ is of integer type `UINT4`
- A higher level scaling `L1` applies to the entire channel, $S_{l_1}$ has floating-point `FP16` datatype
- If we assume many thousands of dimensions in the channel, then $S_{l_1}$ becomes negligible still, but $S_{l_0}$ impacts the bit-width of each value in its respective group of 16 tensor values -> $4.25\ \text{Bit}$

Now, going over to the MX family of more advanced group quantization approaches, we can visualize them like so:
![](./attachments/Pasted%20image%2020260628231141.png)
*Source:* [\[Rouhani et al., 2023\]](https://arxiv.org/abs/2302.08007)

> MX4 replaces an explicit floating-point scale factor $S$ with a shared exponent, since an exponent encodes itself a power-of-two scaling factor.

Remember from last lecture that we can express a floating-point value like so:
$$(-1)^\text{sign} \times (1+ \textbf{Mantissa}) \times 2^{(\textbf{Exponent})-(\textbf{Exponent Bias})}$$
where the *Mantissa expresses precise shape and digits* and the *Exponent acts as a scaler, shifting those digits.* Normally, every floating-point weight has its own sign, exponent and mantissa. MX4 quantization grants each weight:
- $1\ \text{Bit}$ for the Sign
- $2\ \text{Bit}$ for the Mantissa

But then it assumes that the weights roughly live in the same numerical range. So, instead of each weight carrying its own exponent, *one exponent is used for two weights.*
- $2$ weights get scaled by a single shared $1\ \text{Bit}$ Exponent at `L1`; **The Exponent itself encodes the Scaling Factor for quantization** (naively this 1 bit exponent would scale or not scale, but different mappings exist, this is open to the setting)
- At `L2` we group together $16$ such "$1\ \text{Bit}$ for Sign $2\ \text{Bit}$ for Mantissa" weights and those, while already scaled by their local `L1` Exponent Bit, now alltogether get scaled by 8 Exponent Bits on top of that; *like before `L1` is more fine-grained shifting, `L2` is more coarse-grained shifting of weights*
- Effective Bit Width for $MX4$ becomes $3+1/2+8/16 = 4$
	- $3\ \text{Bit}$ for `S1M2` per weight
	- $1/2\ \text{Bit}$ for the `E1` from the lowest-level scaling, amortized across 2 weights
	- $8/16\ \text{Bit}$ for the `E8` from the higher-level scaling, amortized across 16 weights
- *Each weight stores only its sign and mantissa and during dequantization, the shared exponents provide the missing scaling information required to reconstruct an approximate floating-point value*

### Dynamic Range Clipping

Once a model concludes its training, the weights are set and remain fixed. If we wanted to apply Linear Quantization there, we could just scan the tensor and take the actual min/max values from it and calculate $S$ and $Z$ from that, as per this table from last lecture:

|                    | **Asymmetric Quantization (General)**                                                 | **Symmetric Quantization<br>(Common for Weights)**                                  |
| ------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Reconstruction** | $r = (q - Z) S$                                                                       | $r = q S$, because ($Z=0$)                                                          |
| **Scale**          | $S = \frac{r_{\max}-r_{\min}}{q_{\max}-q_{\min}}$                                     | $S = \frac{\|r\|_{max}}{2^{N-1}-1}$                                                 |
| **Zero Point**     | $Z = \operatorname{round}\left(q_{\min}-\dfrac{r_{\min}}{S}\right)$                   | $Z = 0$                                                                             |
| **Quantization**   | $q = \operatorname{clip}\left(\operatorname{round}\left(\dfrac{r}{S}\right)+Z\right)$ | $q = \operatorname{clip}\left(\operatorname{round}\left(\dfrac{r}{S}\right)\right)$ |
| **Typical Use**    | Activations, inputs                                                                   | Weights                                                                             |
| **Reason**         | Supports ranges that are not centered around zero.                                    | Weights are typically approximately zero-centered, so no offset is needed.          |
Now, suppose we also want to *linearly quantize the activations as such* to ease compute and memory demands for inference. The activations emitted form our weights depend on an input, with their numerical range varying across different inputs.

If we want to apply Linear Quantization on Activations, we have to *determine the floating-point range of said activations* bounded by $r_{min}$ and $r_{max}$ first.

![363](./attachments/Pasted%20image%2020260630221801.png)
*Source:* [\[Jacob et al. 2017\]](https://arxiv.org/abs/1712.05877)

*Two ways exist to get a usable $r_{min}$/$r_{max}$ for the dynamic range of the activations, which isn't fixed:*
1) **If we train the model ourselves,** we can collect activations statistics before deploying the model by accumulating an Exponential Moving Average $\hat{r}_{min,\ max} = \alpha \cdot r^{(t)}_{min,\ max} + (1 - \alpha) \cdot \hat{r}_{min,\ max}^{(t-1)}$ with $\alpha$ being the discount factor smoothing the current dynamic range with the previous one (the larger $\alpha$, the more we rely on the most recent range). *This effectively smoothes the dynamic range.*
2) **If the model was trained already,** we can run a few *'calibration' batches* on the `FP32` model and use the mean of the $r_{max,\ min}$ of each sample in the batches and assume that this is representative of the images or inputs we will then face from there on

Note that for the second approach, *we might not want to find $r_{max,\ min}$ to be the exact correct maximum and minimum values.* If we were to set $r_{max}$ to the literal maximum (to capture some outliers), then the scaling factor $S$ would become large, ass $S$ is basically $(\text{range}) / (\text{number of integer levels})$. A large $S$ means each integer step represents a bigger chunk of real-number space.

If we chose the boundary for the minimum a little larger and the boundary for the maximum a little smaller, we would save ourselves some resolution for the now smaller range in between, making the quantization more expressive in the end. *To achieve this, effectively, we would need to find best points to clip the activation distribution.* Usually, we only really need to clip $r_{max}$.

![](./attachments/Pasted%20image%2020260630232617.png)
*Sources:* [\[Banner et al., 2019\]](https://arxiv.org/abs/1810.05723); [\[Sakr et al., 2022\]](https://arxiv.org/abs/2206.06501)

A Laplacian distribution, like the one above, would be nice for this idea, but it is unrealistic to expect for the activations. Instead, we can use the *KL-Divergence:*
$$D_{KL}(P||Q) = \sum_i^NP(x_i)\log \frac{P(x_i)}{Q(x_i)}$$
Given the distributions $P$ and $Q$, $P$ is the distribution of the activations *before* quantization, and $Q$ then gets modeled as the candidate clipping distribution of the activations for best quantization under minimized loss of information.

Looking at an actual activation distribution on the left, we can see its counterpart on the right after *KL-divergence* found a maximum value and clips the activations there. This allows for applying as much resolution as feasible (according to KL-divergence) to as much of the activation range as necessary.

![](./attachments/Pasted%20image%2020260630232334.png)
*Source:* [\[Szymon Migacz, 2017\]](https://www.cse.iitd.ac.in/~rijurekha/course/tensorrt.pdf)

### Rounding

Coming back to weights again, rounding weights each to nearest value *is not optimal for the behavior of the resulting tensor.* Instead, we have to round in such a way that *the reconstructed activation is as close to the original as possible.*

> Instead of $\left\lfloor w \right\rceil$ (rounding up or down depending on what's nearer to $w$) we do $\{\left\lfloor w \right\rfloor, \left\lceil w \right\rceil \}$ to best retain the original activation.

We could also go the *AdaRound* route. AdaRound *learns for each weight* whether rounding up or down gives the smallest error in the layer's output, given some input data. It achieves this by adding a $\delta \in [0,\ 1]$ so that $\tilde{w} = \lfloor \lfloor w \rfloor + \delta \rceil$. *Written out, we find:*
$$\text{argmin}_V ||W_x - \tilde{W}_x||^2_F + \lambda f_{reg}(V)\ \ \rightarrow \ \ \text{argmin}_V ||W_x - \lfloor \lfloor W\rfloor + h(V)\rceil x||^2_F + \lambda f_{reg}(V)$$
- $x$ is the *calibration input* (the sample activations fed into the layer as input), the goal is to make the layer output $\tilde{W}x$ close to $W x$
- $V$ is a *learnable optimization parameter* (same shape as the weights $W$), determining whether each weight should round up or down
- $h(V)$ maps each element of $V$ into $(0,1)$, usually with a rectified sigmoid. This produces a "soft" rounding offset:  
    - if $h(V) \lt 0.5$, the weight rounds down
    - if $h(V) \gt 0.5$, it rounds up
- $f_{\text{reg}}(V)$ encourages $h(V)$ to become close to $0$ or $1$, so final rounding decisions are binary

> AdaRound learns, for each weight, whether it should be rounded up or down based on representative input activations (a calibration set), so as to minimize the difference between the original and quantized layer outputs.

We've now seen quite a set of quantization techniques to meaningfully preserve accuracy.
However, even despite the Activation- and Weight-level quantization being applied on GoogleNet without issue, they still cause significant accuracy drops in smaller networks, see MobileNet.
*This is presumably due to their inherently lower representational capacity.*

![](./attachments/Pasted%20image%2020260702014329.png)

> How should we further improve the performance of quantized models in such a way that even smaller models can retain their accuracy?

## Quantization-Aware Training (QAT)

In the [last lecture](Research/AI/MIT%20EfficientML.ai%20Fall%20'24/05%20-%20Quantization%20I.md) we already had introduced K-Means-based Quantization:
![521](./attachments/Pasted%20image%2020260627003102.png)
*Source:* [\[Han, et al. 2016\]](https://arxiv.org/abs/1510.00149)

During Quantization-Aware Training (QAT), the network *uses quantized weights for its computations, but updates a full-precision copy of these weights, which then get quanitzed again as training progresses.* This is because many gradient updates can be too small to be represented in low precision and would otherwise be lost. *After each update, the full-precision weights are quantized again* for the next training step. Once training is complete, the full-precision copy is discarded, leaving only the quantized weights for inference.

With Linear Quantization, both the weights and activations are represented in low precision.
QAT is a method to train linearly quantized models. While it doesn't produce weights that can be linearly quantized, it trains the model *while it is already in a linearly quantized state and updated in that realm* with the help of the full-precision 'proxy weights' actually receiving the updates.

![697](./attachments/Pasted%20image%2020260702023502.png)

*We can also call this approach "Fake Quantization"* as everything within the net still directly depends on the full-precision weights evolving. However, the net itself operates in fully quantized fashion from the start. *The weights $W$ thus inform their quantized versions $Q(W)$ throughout.*

Quantization is a discrete operation, so many nearby full-precision values map to the same quantized value. As a result, small changes to a weight or the activation often produce no change in its quantized counterpart (blue staircase below), so the derivative of that becomes zero almost everywhere:
$$\frac{\partial Q(W)}{\partial W} = 0$$
![](./attachments/Pasted%20image%2020260702024224.png)

Without gradients, gradient descent cannot update the full-precision weights. To avoid this, QAT uses the *Straight-Through Estimator (STE).* During the forward pass, quantization is performed normally. However, during the backward pass, we pretend that the weight quantization was the identity function, passing gradients through unchanged:
$$\text{gradient}_W = \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Q(W)}$$
Although the forward pass uses quantized weights, *during backpropagation we "jump over" the quantization step and treat it as the identity function, as if $Q(W)=W$.* This allows gradients to flow directly to the underlying learnable full-precision weights $W$. The quantized weights are then recomputed from the updated full-precision weights for every iteration.

*This approach can be extended to the activation quantization just the same,* so that during the backward pass, both quantization operations are treated as the identity so gradients can flow through them.
$$\text{gradient}_W = \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Q(W)}, \quad  
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial Q(x)}$$
**QAT recovers what previous had gone missing for smaller models like MobileNet to be able to retain accuracy meaningfully:**
![](./attachments/Pasted%20image%2020260702031631.png)

> **As a rule of thumb:** The larger the model, the less likely it is for quantization-aware training to really be necessary. *For example, it is sufficient for LLMs to receive post-training quantization.*

**Now, let's push the quantization stack to an extreme.**

## Binary and Ternary Quantization

Suppose we could quantize to such an extent that we could only retain binary weights and bit operations for computation. **Can we push quantization precision to only 1 bit?**

![](./attachments/Pasted%20image%2020260702123542.png)
*Source:* [\[Courbariaux et al., 2015\]](https://arxiv.org/abs/1511.00363)

Representing weights as binary values (±1) eliminates the need for general multiplication. *Each weight now only adds or subtracts its input,* which reduces both computation and memory requirements.

Two ways exist for attaining such a binary weight configuration:

**Deterministic Binarization:**
- This basically applies the sign function: If the weight is positive or $0$, it becomes $+1$, otherwise $-1$
- Seems reasonable, but might in the context be insensitive, degrading accuracy

**Stochastic Binarization:**
- Instead of rounding a weight outright to the nearest value of $+1$ and $−1$, we choose between the two
- The probability of choosing each value depends on the original weight
	- If the real-valued weight $r$ is close to $+1$, it is *very likely to become $+1$*
	- If $r$ is close to $−1$, it is *very likely to become $−1$*
	- If $r$ is near $0$, either outcome is possible, with roughly equal probability

This decision probability is computed using the *hard sigmoid function* $\sigma(r)=\min\left(\max\left(\frac{r+1}{2},0\right),1\right)$, which maps any real number to a probability between $0$ and $1$. The quantized weight is then sampled as
$$q=  
\begin{cases}  
+1 & \text{with probability } \sigma(r),\\  
-1 & \text{with probability } 1-\sigma(r)  
\end{cases}$$
It may seem counterintuitive, *noising the deterministic binarization approach,* but the *randomness actually helps preserve information on average.* Instead of always forcing borderline values to the same binary value, they get distributed over the course of many training steps, which can help retain nuance and reduce quantization error to improve training performance.

 However, this requires generating random numbers every time weights are quantized, making the hardware implementation more complex and energy-intensive.

> This idea alone will work, but it will come at quite a large accuracy drop due to a high induced quantization error.

![](./attachments/Pasted%20image%2020260702135507.png)

We can see that something called *Binary Weight Network (BWN)* pretty much mitigates this drop entirely, though.

### Binary Weight Network (BWN)

We can go back and combine our binary weights with a single floating-point scaling factor, akin to what we did with [Linear Quantization](Research/AI/MIT%20EfficientML.ai%20Fall%20'24/05%20-%20Quantization%20I.md). We calculate this scaling factor differently now, though. Given the original weights, we calculate the scaling factor $\alpha$ like so:
$$\alpha = \frac{1}{n}||W||_1$$
With $n$ being the weight count and $||W||_1$ being the L1-Norm, i.e. the sum of the absolute values of the original weights. *Even though we only see the quantization error drop minimally,* $||W-W^B||^2_F = 9.24$, we find that the accuracy drop is nearly completely mitigated now.

![](./attachments/Pasted%20image%2020260702142925.png)
*Source:* [\[Rastegari et al., ECCV 2016\]](https://arxiv.org/abs/1603.05279)

>[!Question] How is it possible that the accuracy delta changed from $-21.2\%$ to $+0.2\%$ while quantization error only improved from $9.28$ to $9.24$?
>This is possible because our Frobenius quantization error is not a good predictor of network accuracy. It measures only the overall weight difference by treating all weights as equally important. The scaling factor preserves information about the magnitude of each layer's weights, making the network's activations much closer to those of the full-precision model.

### Binary Quantizing Both Weights and Activations

We could apply different logical frameworks for how to combine the weights with the activations. First, we could go the classical route with $Y = WX$. We could also make it $XNOR(b_w,b_x)$:

![](./attachments/Pasted%20image%2020260702144739.png)

$XNOR$ is much easier to implement in hardware, but we need to implement $Y=WX$ to build a network layer. *The trick is that we can substitute $Y=WX$ by using the $XNOR$ operation,* multiplying its result $\times 2$ and then adding a $-1$ per $XNOR$ calculation to build the final sum. For example, $Y = WX$ with $W=1,\ X = -1$ can be expressed like so: 

$XNOR(1,0) = 0,\ 0 \times 2 = 0,\ 0 + (-1) = -1$

Note that the $\times 2$, while a multiplication, is just a very easily implementable bit-shift operation. If we now scale this idea to apply to the vector multiplication of weights and (prior layer's) activations, we get this, summing the $XNOR$ results and the amount of times we applied $XNOR$, respectively:
![692](./attachments/Pasted%20image%2020260702150428.png)

Because calculations $1+0+0+0$ effectively have us count the amount of $1$'s from the $XNOR$ results, instead of summing explicitly, we can apply the function $\text{popcount}()$ to return this amount of $1$'s.

![693](./attachments/Pasted%20image%2020260702151840.png)

This yields us significant boosts in computation:
![](./attachments/Pasted%20image%2020260702152651.png)

BWN is good, but the activations are still 32 bit and we need a scale on top of the binarized weights.
XNOR-Net kind of recovers from the BNN's desastrous accuracy loss, but its still at a bad $-12.4\%$ accuracy. *Can we improve over XNOR-Net while not becoming as demanding as BWN?*

### Ternary Weight Networks (TWN)

Let's allow for three states $[-1,\ 0,\ 1]$ and assign states to represent the weights like so:$$q=  
\begin{cases}  
r_t & r \gt \Delta,\\  
0 & |r| \leq \Delta\\
-r_t & r \lt -\Delta
\end{cases}$$ with $\Delta$ being some threshold $\Delta = 0.7 \times \mathbb{E} (|r|),\ r_t = \mathbb{E}_{|r| > \Delta}(|r|)$, with $\mathbb{E}(\dots)$ calculating the average. (The $0.7$ is empirical)

Akin to [Binary Weight Networks](#Binary%20Weight%20Network%20(BWN)), we derive a scaling factor, but only have non-zero weights of the ternary weight matrix affect it like so:
![](./attachments/Pasted%20image%2020260703130009.png)

![397](./attachments/Pasted%20image%2020260703130152.png)
*Source:* [\[Li et al., 2016\]](https://arxiv.org/abs/1605.04711)

### Trained Ternary Quantization (TTQ)

Departing from the be strict bit-level ternary view a little, we can also make $r_t$ and $-r_t$ distinct and learnable parameters $w_p$ and $w_n$ and then have those replace the original weights like so:$$q=  
\begin{cases}  
w_p & r \gt \Delta,\\  
0 & |r| \leq \Delta\\
-w_n & r \lt -\Delta
\end{cases}$$We build the TTQ on top of the notions of  the TWN, so to say:
![](./attachments/Pasted%20image%2020260703130957.png)
First, TTQ uses the same ternarization rule as TWN, where the thresholding step is the same.
However, *based on this thresholding, TTQ doesn't assign $[-1,0,1]$, but $[-w_n, 0, w_p]$. These two learnable parameters also replace the use of an additional scaling factor.* The error induced by the resulting weight matrix then serves as basis for optimization of the latent full-precision weights as well as both $w_p$ and $w_n$

![](./attachments/Pasted%20image%2020260704164644.png)

> After all, TTQ is only a minor improvement over standard 2-bit TWN.

![](./attachments/Pasted%20image%2020260704165205.png)

> Quantization around 4-bit is often considered a practical sweet spot (i.e. affording $15$ instead of $3$ discrete states). Reducing precision lowers memory usage roughly linearly, but it can degrade compute efficiency and model expressiveness quadratically. Since memory is typically the bottleneck compared to compute, overly aggressive quantization hurts the performance more than it helps. In practice, it’s usually better not to push any single technique (like quantization) to its limit, but to combine it with other methods, e.g. pruning, distillation, and neural architecture search.

## Automatic Mixed-Precision Quantization

The approaches discussed so far could naively be applied across all layers of a network. Different layers may have different sensitivities to such efforts, though. The more sensitive a layer is, the more bits we should assign to it, and vice versa.

Suppose a compute budget allowing for a network's layers being at most $8\ \text{Bit}$ for weights and another $8\ \text{Bit}$ for the activations. If we now quantize per layer, according to its requirements, we thus effectively have to choose from $8\times8$ design configurations for each of $n$ layers (total of $64^n$ possible designs).

We can optimize in this quite large search space for example by using RL via an actor-critic setup:
![572](./attachments/Pasted%20image%2020260704175646.png)
*Source:* [\[Wang et al., 2019\]](https://arxiv.org/abs/1811.08886)

Mapping an actor's configuration choice to the hardware, we can measure latency and energy consumption and feed that back to the critic to improve the policy.

> This HAQ approach is overcomplicated, but as rule of thumb, we get that extra edge of efficiency by addressing different layers according to their specific needs.

*This approach already kind of hints at the next lecture.* HAQ can support multiple different objectives in its mixed precision optimization: Model size, Latency, Energy demand. This support for different objectives will also become crucial in the next topic, neural architecture search.
