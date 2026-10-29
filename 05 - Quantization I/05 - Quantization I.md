# 05 - Quantization I

- [Quick Recap](#quick-recap)
- [Numeric Data Types](#numeric-data-types)
  - [Integer](#integer)
  - [Fixed-Point Numbers](#fixed-point-numbers)
  - [Floating-Point Numbers](#floating-point-numbers)
  - [Subnormal Values](#subnormal-values)
- [Quantization](#quantization)
- [K-Means-based Quantization](#k-means-based-quantization)
  - [Finetuning a Quantized Model](#finetuning-a-quantized-model)
  - [Huffman Coding](#huffman-coding)
- [Linear Quantization](#linear-quantization)
  - [Linear Quantized Matrix Multiplication](#linear-quantized-matrix-multiplication)
  - [Linear Quantized Fully-Connected Layer](#linear-quantized-fully-connected-layer)
  - [Linear Quantized Convolutional Layer](#linear-quantized-convolutional-layer)
  - [How Does INT8 Linear Quantization Compare to Floating Point Weights?](#how-does-int8-linear-quantization-compare-to-floating-point-weights)
  - [Summary of Linear Quantization](#summary-of-linear-quantization)

## Quick Recap

In the last lecture, we learned that pruning as a tool would help reduce the amount of parameters present in a neural network. These techniques alone brought speedups and energy savings.

*The total size of a neural network very roughly amounts to:* $\text{number of parameters} \times \text{number of bits per parameter}$

>[!NOTE]
> With Quantization, we try to reduce the number of bits necessary to express each parameter.
> Operations with lower bit-width are more energy-efficient and less memory-demanding. A lower demand in memory also reduces the demand for costly data movement on top of that.

## Numeric Data Types

In order to understand how different numeric data types may increase or decrease a network's efficiency, we have to look at how said data types are represented in memory.

### Integer

Given a memory bank with $n$ bits.

An *unsigned integer* uses the full $n$ bits to represent the numeric value. For the below case, this is $49$. The total expressable range of possible unsigned values is $[0,\ 2^n-1]$.
Now, a *signed integer* requires us to reserve $1$ bit (say this is the highest bit) to express whether a value is positive or negative. This bit has no effect on the value, meaning that *$[0000\ 000]$ and $[1000\ 000]$ would both represent $0$, which is an overlap and a little wasteful.*

Signed integers may also be expressed through the two's complement, which eliminates this overlap.
The highest bit $n-1$ represents $-2^{n -1}$, a negative value, while the other bits still remain positive in their meaning.

<img src="./attachments/Pasted%20image%2020260521214120.png" width="250" />

>[!CAUTION]
>**Question:** Why exactly does the two's complement resolve the problem of overlap?

>[!TIP]
>**Answer:** With the two's complement, the highest bit *actively contributes to the value* through its negative magnitude. On its own, we find $[1000\ 0000] = -2^7 = -128$.<br>
> Crucially, $[1000\ 0000] = -128 != [0000\ 0000] = 0$ with the two's complement. The highest magnitude value expressable with $7$ bits is $2^7 - 1 = 127$, meaning $[1111\ 1111] = -1$. Thus, every possible bit pattern now maps to a unique numeric value in the range $[-2^{n-1},\ 2^{n-1}-1]$.<br>*We eliminated the overlap.*

### Fixed-Point Numbers

Suppose again a memory bank of $n$ bits. We again use the two's complement logic.
With the fixed-point paradigm, we now split our $n$ bits into two half, the first being additionally reduced to $n/2 - 1$ by one bit for the two's complement.
Regarding the fixed point number, the remaining first part represents the integer, while the second part represents the fraction after the decimal point.

<img src="./attachments/Pasted%20image%2020260521221144.png" width="420" />

*There are two ways to read the fixed-point representation:*
The first representation uses the decimal point as the reference. From this point, bit values increase to the left into positive range in the integer part and decrease into negative range to the right in the fractional part. By contrast, the lower representation starts from $2^0$, but it then shifts this value to the right by $4$ (equivalent to multiplying by $2^{-4}$). The $4$ comes from the amount of bits representing the fraction.

>[!NOTE]
> The second visualization in the above image is the important one: *Fixed-point* representation stores a number as a single two's complement-represented integer. It is then interpreted by being scaled with a *fixed factor*, meaning that we assume a fixed position of the binary point for *all fixed-point values*.

### Floating-Point Numbers

>[!NOTE]
> Floating-point numbers can express a larger dynamic range. To do so, they apply a fundamentally different bit representation compared to the fixed-point logic. Where before we pretty directly expressed the numeric value and then shifted it, we now express the components to make up the value itself.

According to IEEE 754, we can define a `float32` number with $1$ sign bit, $8$ bits for an exponent (not an integer component) and $23$ bits for the so-called fraction. *With floating-point numbers, the fraction is often referred to as significant or mantissa.*

<img src="./attachments/Pasted%20image%2020260522090554.png" width="600" />

The difference to fixed-point representation becomes clear when looking at the relationships of these floating-point bit groups:
$$(-1)^\text{sign} \times (1+ \textbf{Fraction}) \times 2^{(\textbf{Exponent})-(\textbf{Exponent Bias})}$$
- $(-1)^\text{sign}$:  You might have expected the two's complement to make an appearance here, but IEEE 754 does not use that. The sign is stored separately.
- $(1 + \textbf{Fraction})$: The fraction is flat out incremented by $1$, always.
- $2^{(\textbf{Exponent}) - (\textbf{Exponent Bias})}$: We have our sign bit, signifying the sign of the entire resulting floating-point number. It is possible for the final floating-point value to result from a negative exponent and a positive fraction. Thanks to this option, floating-point can represent both very large and very small values alike. 
	- The $\text{Exponent Bias}$ is calculated using $(\text{Exponent Bits} - 1)$ as exponent, then subtracted by $1$. For our $8$ bit exponent, this amounts to an $\text{Exponent Bias}$ of $2^{8-1}-1 = 127$
	- *Reason for Exponent Bias:* The exponent is stored with a removable bias so that both positive and negative exponents can be represented using an unsigned bit pattern.

We can now express $0.265625$ through $(-1)^0 \times (1 + 0.0625) \times 2^{125-127}$ and represent that like so:<br>
<img src="./attachments/Pasted%20image%2020260522102155.png" width="600" />

We are forced to use this $(1+\textbf{Fraction})$ term. **But now, how would we represent $0$?**

### Subnormal Values

To represent $0$, we define an entire category of *subnormal values*. This category covers all those values where the $\text{Exponent Bits}$ are all $0$. If this is the case, we *artificially set the $\text{Exponent} = 1$ in the bias correction term* and then calculate the final number like so:
$$(-1)^\text{sign} \times \textbf{Fraction} \times 2^{\textbf{ 1}-127}$$
The sign bit remains irrelevant to whether or not we express a $0$ in this fashion:<br>
<img src="./attachments/Pasted%20image%2020260522104020.png" width="480" />

Given the exponent has to remain $0$, the *largest positive subnormal value* we can express now becomes solely governed by the fraction:<br>
<img src="./attachments/Pasted%20image%2020260522124138.png" width="480" />

This is still smaller than the *smallest normal value* (where the exponent term has to be at least $1$):<br>
<img src="./attachments/Pasted%20image%2020260522125308.png" width="480" />

Note that the *largest positive and negative normal values* represent $+\infty$ and $-\infty$:<br>
<img src="./attachments/Pasted%20image%2020260524170525.png" width="480" />

>[!NOTE]
> If the exponent bits are all $1$, but there are any non-zero entries in the fraction bits, the resulting expression represents $\text{NaN}$. As this would be the interpretation for any combination of non-zero values present, this mapping to $\text{NaN}$ is quite wasteful.

*To summarize the `float32` representation scheme:*<br>
<img src="./attachments/Pasted%20image%2020260528111641.png" width="700" />

A simple way to reduce complexity and thus decrease compute and memory throughput demands would be to reduce the bit precision of the `float` value. Note that *reducing the exponent means reducing the range, while reducing the fraction means reducing the precision within that range, the resolution if you will.*

<img src="./attachments/Pasted%20image%2020260528112903.png" width="800" />

These `FP8` types introduced by NVIDIA can be found in the H100 series GPUs.
More recently, with the Blackwell series (B100 etc.) NVIDIA went even further and introduced `FP4` and `INT4`:<br>
<img src="./attachments/Pasted%20image%2020260528113514.png" width="800" />

## Quantization

>[!IMPORTANT]
>Quantization is the process of constraining an input from a continuous or otherwise large set of values to some discrete (smaller) set.
> The goal of quantization is to *save memory and meaning by reducing the need for precision in a way that minimizes the representational error induced by that reduction.*

<img src="./attachments/Pasted%20image%2020260528114135.png" width="450" />

## K-Means-based Quantization

Given a weight matrix, we approximate the weights by replacing them with integers referencing entries in a codebook. *The codebook then only holds a few floating-point values.*

<img src="./attachments/Pasted%20image%2020260627000001.png" width="800" /><br>
*Source:* [\[Han, et al. 2016\]](https://arxiv.org/abs/1510.00149)

We *save storage for the weight matrix* by using integers to reference a set of rounded centroid floats. This *doesn't save us compute,* however. We still perform floating-point arithmetic with all referenced float values.

Rounding of the float values avoids the codebook growing too large, which in turn improves storage savings. It also introduces *quantization errors* in the reconstructed weights.

- The *original weights* make up $32\ \text{Bit} \cdot 16\ \text{Weights} = 512\ \text{Bit}$, *generally $32\ \text{Bit} \cdot M\ \text{Weights}$*
- The *cluster indexes* make up $2\ \text{Bit} \cdot 16\ \text{Weights} = 32\ \text{Bit}$, *generally $N\ \text{Bit} \cdot M$*
- The *codebook adds to the cluster indexes* another $32\ \text{Bit} \cdot 4\ \text{Centroids} = 128\ \text{Bit}$, *generally $32\ \text{Bit} \cdot 2^N$*
- We arrive at a *total of $160\ \text{Bit}$ of storage for the K-Means-based Quantization*

### Finetuning A Quantized Model

Suppose that backprop produces a gradient matrix. The gradients of multiple weights associated with the same centroid now have to get grouped together to affect that centroid:

<img src="./attachments/Pasted%20image%2020260627003102.png" width="500" /><br>
*Source:* [\[Han, et al. 2016\]](https://arxiv.org/abs/1510.00149)

**This above image is a bit misleading.** During the forward pass, *we do not apply the upper left matrix, but its quantized version.* That version 'causes' the gradient matrix.

To have this matrix now properly inform the quantized representation, we have to *group and sum those gradients that share positions with the same cluster index.* It is those gradient sums that then get applied to the centroids.

> [!CAUTION]
> **Question:** Why are we summing the gradient per centroid, and not average them?

> [!TIP]
> **Answer:** It is possible to average them, but we argue that the more weight indices want to affect a centroid, the larger the nudge applied to that weight should be, and summing does this better than averaging.

<img src="./attachments/Pasted%20image%2020260627004315.png" width="650" /><br>
*Source:* [\[Han, et al. 2016\]](https://arxiv.org/abs/1510.00149)

Based on retained accuracy, pruning allows for slightly higher better model compression than quantization, but applying both together really makes a difference, allowing for $\sim 3\%$ remaining model size while maintaining accuracy.

>[!CAUTION]
>**Question:** Should we prune or should we quantize first?

>[!TIP]
> **Answer:** Prune first, eliminate redunant weights, then quantize the remaining weights more effectively. This reduces quantization workload while enabling higher quality centroids to form (better approximation possible by the centroids for less weights to represent)

<img src="./attachments/Pasted%20image%2020260627010542.png" width="650" /><br>
*Source:* [\[Han, et al. 2016\]](https://arxiv.org/abs/1510.00149)

Retraining after pruning and quantization, for which we just saw the gradient attribution mechanism above, can further improve the retained accuracy.<br>
Through this, the centroids may shift slightly to "settle into a most task-appropriate expressive state".

We introduced K-Means-based Quantization with an example of 2-bit indices ($k=4$).<br>
This actually is a *good choice for fully-connected layers,* but *for convolutional layers, you want to up that to 4 bits ($k = 16$)* to best retain accuracy and storage gains alike:<br>
<img src="./attachments/Pasted%20image%2020260627011526.png" width="750" /><br>
*Source:* [\[Han, et al. 2016\]](https://arxiv.org/abs/1510.00149)

### Huffman Coding

After having applied quantization, every weight is replaced by a cluster index. These indices are not equally common. Huffman Coding leverages this by changing centroid indices to have variable-length binary codes with the length depending on their centroid's frequecy of appearance. *Frequent centroids receive shorter Huffman codes for their indices, while rare ones receive longer codes for their indices.*

>[!NOTE]
> Huffman Coding gets applied last to further reduce the storage cost of the quantized model by exploiting the uneven frequency of centroid indices for encoding most common ones with fewer bits.

<img src="./attachments/Pasted%20image%2020260627012029.png" width="550" /><br>
*Source:* [\[Han, et al. 2016\]](https://arxiv.org/abs/1510.00149)

| Centroid | Frequency   | Huffman code<br>(Centroid Index) | Centroid<br>(Example) |
| -------- | ----------- | -------------------------------- | --------------------- |
| $c_0$    | very common | 0                                | -0.5                  |
| $c_1$    | common      | 10                               | 2.7                   |
| $c_2$    | rare        | 110                              | 1.2                   |
| $c_3$    | very rare   | 111                              | 5.8                   |

On average, variable-length Huffman coding reduces the storage required for the cluster indices, which allows to further compress the weight matrix without changing anything about the represented centroids.
Adding this to pruning and quantization brings about a $40 \times$ compression of LeNet-300, for example, while retaining or slightly gaining accuracy.

<img src="./attachments/Pasted%20image%2020260627014038.png" width="650" /><br>

Different models of different architectures, sizes and purposes allow for different compression ratios to be applied while the accuracy still remains very much unharmed:

<img src="./attachments/Pasted%20image%2020260924230230.png" width="550" /><br>
*Source:* [\[Han, et al. 2016\]](https://arxiv.org/abs/1510.00149)

## Linear Quantization

Where K-Means-based Quantization used integer weights and floating-point arithmetic, *linear quantization uses integer weights and integer arithmetic.* This is very widely used in AI applications running locally on phones or IoT devices.

<img src="./attachments/Pasted%20image%2020260627152955.png" width="800" /><br>
*Source:* [\[Jacob et al., 2018\]](https://arxiv.org/abs/1712.05877)

We can map each float weight to a small integer via a ruler-like linear mapping, defined by two parameters, the Zero Point $Z$ and the Scale $S$:
$$r = (q - Z) \times S$$
- $S$ expresses how much real-world float value one integer step represents
- $Z$ tells which integer tick lands on float zero

Suppose a field in the original matrix holds the value $0$. The goal for $Z$ is to represent this value $r=0$ from the original weight matrix exactly when combined with $q$ via $(q-Z)$. $S$ then, beyond that, improves the dynamic range of $(q-Z)$:

<img src="./attachments/Pasted%20image%2020260627154452.png" width="550" /><br>
*Source:* [\[Jacob et al., 2018\]](https://arxiv.org/abs/1712.05877)

*We again use the two's complement encoding,* so $q$ may take on integer values in the following ranges:

| Bit Width | $q_{min}$    | $q_{max}$     |
| --------- | ------------ | ------------- |
| **2**     | $-2$         | $1$           |
| **3**     | $-4$         | $3$           |
| **4**     | $-8$         | $7$           |
| **N**     | $-2^{(N-1)}$ | $2^{(N-1)}-1$ |

Given the bit width, $q_{min}$ and $q_{max}$ are fixed. They have to be able to capture and map to $r_{min}$ and $r_{max}$, which we find by looking at the actual min and max of our float weights. The scaling factor $S$ works as the bridge between those two ranges:
$$r_{max} = S (q_{max} - Z)$$
$$r_{min} = S(q_{min} - Z)$$
This leads us to finding the *scaling factor $S$* from the ratio of the dynamic range of $r$ and the dynamic range of $q$:

$$S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}$$

Remember, $r$ is *not the reconstructed weight matrix,* but the *original weight matrix* we want to quantize:<br>
<img src="./attachments/Pasted%20image%2020260627220955.png" width="550" />

To now determine the *zero point integer $Z$*, we calculate:

$$Z = \text{round}\left( q_{min} - \frac{r_{min}}{S} \right)$$

<img src="./attachments/Pasted%20image%2020260627221809.png" width="550" />

### Linear Quantized Matrix Multiplication

>[!NOTE]
> How can we use the quantized matrix version for matrix multiplication?

Remember, this is what we will work with now, especially the *quantized weights:*<br>
<img src="./attachments/Pasted%20image%2020260627152955.png" width="800" />

The goal is to express general matrix multiplication $Y = WX$ using only integer arithmetic, and then have *floats appear only as a single scale factor applied once to the final result.* 

To now do this, first, we can substitute $r = (q - Z) \times S$ for $Y$, $W$, and $X$:
$$\underbrace{(q_Y - Z_Y) \times S_Y}_{Y} = \underbrace{(q_W - Z_W) \times S_W}_W \times \underbrace{(q_X - Z_X) \times S_X}_X$$
Rearranging this yields:
$$q_Y = \frac{S_W S_X}{S_Y} \cdot \left( q_W q_X - Z_W q_X \underbrace{- Z_X q_W + Z_W Z_X}_\text{Precomputable} \right) + Z_Y$$

The scale factor $\frac{S_W S_X}{S_Y}$ is the one remaining float operation in an otherwise integer computation. *Empirically, \[Jacob et al., CVPR 2018\] found that this scale factor is always within the interval $(0,1)$.* Therefore we can simplify its calculation like so:
$$\frac{S_W S_X}{S_Y} = 2^{-n}M_0$$
where $M_0 \in [0.5,\ 1)$. This means $M_0$ can be stored and multiplied as a fixed-point integer, i.e. an integer whose bits are simply interpreted as a fraction. Integer multiplication units are standard and cheap on all hardware, including mobile chips. Therefore the entire scale factor collapses into one right-shift plus one integer multiply, with no float unit needed anywhere. We effectively further reduced the performance impact of this one necessary float scaling factor.

$Z_W$ is the integer that represents the *zero point*, i.e. zero for the represented weights.
If the weights we want to quantize are normally distributed around zero, then this *zero point* sits right in the middle of its weight distribution at $0$. We then effectively lose nothing by *outright (and correctly in this context) setting $Z_W = 0$.*

Where we previously had the more general left configuration, normally distributed weights around zero have us encounter the right configuration:<br>
<img src="./attachments/Pasted%20image%2020260627224405.png" width="750" />

In case we have this configuration on the right, we can simplify

$$S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}$$

to the expression

$$S = \frac{|r|_{max}}{2^{N-1}-1}$$

which is simpler to calculate. Then, under the assumption that $Z_W = 0$, calculating $q_Y$ becomes:

$$q_Y = \frac{S_W S_X}{S_Y} \cdot \left( q_W q_X\underbrace{- Z_X q_W}_\text{Precomputable} \right) + Z_Y$$

Computation-wise, the heavy/expensive lifting is done by $q_W q_X$, which is now pure integer arithmetic.
Remember, the scale ratio $\frac{S_W S_X}{S_Y}$ is just one small multiplication applied once to the result at the end.

### Linear Quantized Fully-Connected Layer

*We just expressed $Y = WX$ via linear quantization.*
This is not far off from $Y = WX + b$, the default fuly-connected layer.

**We too can rewrite accordingly:**
$$S_Y \cdot (q_Y - Z_Y) = S_W \cdot (q_W - Z_W) \cdot S_X \cdot (q_X - Z_X) + S_b \cdot (q_b - Z_b)$$
**We can now make these assumptions:**
- $Z_W = 0$: Weights are zero-centered, no shift is needed
- $Z_b = 0$: The bias $b$ is just a free parameter chosen during training, so we can constrain it such that its quantization requires no offset correction.
- $S_b = S_W S_X$: The bias is added to the result of $WX$, so it needs to be on the same scale as $WX$.
            This keeps everything in the same units so the addition works.
- $q_{Bias} = q_b - Z_X q_W$: Just rewriting this as one term, absorbing the remaining $Z_X q_W$ correction term from the matrix multiplication into the bias

**Re-arranging this under these yields:**
$$q_Y = \frac{S_W S_X}{S_Y} \cdot \left( q_W q_X + q_{Bias} \right) + Z_Y$$
Both $q_b$ and $q_{bias}$ are 32 bits. Again, $q_W q_X$ does the computational heavy lifting.
Again, we scale in the end by $(S_W S_X) / (S_Y)$.

### Linear Quantized Convolutional Layer

Generally expressed as $Y = \text{Conv}(W, X) + b$, under the familiar assumptions that $Z_W = 0,\ Z_b = 0,\ S_b = S_WS_X,\ q_{Bias} = q_b - \text{Conv}(q_W, Z_X)$ we can say that:
$$q_Y = \frac{S_W S_X}{S_Y} \cdot \left( \text{Conv}(q_W,\ q_X) + q_{Bias} \right) + Z_Y$$

*The structure is identical to the fully-connected case.*
The convolution itself runs on integers, and the scale factor is applied once afterward.

*To show how this is achieved step-by-step, the computation graph for the LQConv Layer looks like this:*<br>
<img src="./attachments/Pasted%20image%2020260627233027.png" width="450" />

### How does INT8 Linear Quantization Compare to Floating Point Weights?

<img src="./attachments/Pasted%20image%2020260627233318.png" width="450" /><br>
*Source:* [\[Jacob et al., 2018\]](https://arxiv.org/abs/1712.05877)

We can see that switching to Int8 Linear Quantization provides either higher accuracy at roughly same latency or same accuracy at lower latency.

### Summary of Linear Quantization

<img src="./attachments/Pasted%20image%2020260627234404.png" width="750" /><br>

Where **K-Means-based Quantization** only saved storage, but no computation, **Linear Quantization** *wants to compute passes through neural network layers using integer arithmetic only.* Every float value $r$, i.e. the weights, inputs, outputs, gets mapped to an integer $q$ via: $$r = (q - Z) \times S$$ where $S$ and $Z$ are chosen to cover the actual range of the floats as faithfully as (feasibly) possible: 

$$S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}} \qquad Z = \text{round}\left(q_{min} - \frac{r_{min}}{S}\right)$$

Once we substitute this mapping into $Y = WX$, the expensive part, multiplying weights by inputs, becomes pure integer arithmetic in $q_W q_X$. *The only float operation left is a single scale factor applied once to the result, and even that collapses into a cheap right-shift plus one integer multiply:* 

$$q_Y = \frac{S_W S_X}{S_Y} \cdot \left(q_W q_X + \underbrace{q_{Bias}}_\text{Precomputed}\right) + Z_Y \qquad \frac{S_W S_X}{S_Y} = 2^{-n}M_0$$

*The simplifying assumptions clean up the remaining terms:*
- $Z_W = 0$ because weights are zero-centered so no offset correction is needed
- $q_{Bias}$ absorbs all remaining correction terms ($q_b$ and $Z_X q_W$) into a single constant that depends only on weights and quantization parameters, computed once ahead of time and stored rather than recomputed at every inference pass
- $S_b = S_W S_X$ forces the bias onto the same scale as $WX$ so the addition is consistent

Convolutional and fully-connected layers follow this same pattern, making the structure of their final formulas identical.

|                    | **Asymmetric Quantization (General)**                                                 | **Symmetric Quantization<br>(Common for Weights)**                                  |
| ------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Reconstruction** | $r = (q - Z) S$                                                                       | $r = q S$, because ($Z=0$)                                                          |
| **Scale**          | $S = \frac{r_{\max}-r_{\min}}{q_{\max}-q_{\min}}$                                     | $S = \frac{\|r\|_{max}}{2^{N-1}-1}$                                                 |
| **Zero Point**     | $Z = \text{round}\left(q_{\min}-\dfrac{r_{\min}}{S}\right)$                           | $Z = 0$                                                                             |
| **Quantization**   | $q = \text{clip}\left(\text{round}\left(\dfrac{r}{S}\right)+Z\right)$                 | $q = \text{clip}\left(\text{round}\left(\dfrac{r}{S}\right)\right)$                 |
| **Typical Use**    | Activations, inputs                                                                   | Weights                                                                             |
| **Reason**         | Supports ranges that are not centered around zero.                                    | Weights are typically approximately zero-centered, so no offset is needed.          |

**The choice depends on the data distribution:** If values are approximately zero-centered, use *symmetric* quantization.
Otherwise, use *asymmetric* quantization to better utilize the available integer range.

>[!NOTE]
> Integer arithmetic is a lot faster and more energy-efficient than float arithmetic, especially on mobile hardware. *In the next lecture, we will see binary and ternary quantization to compress models even further.*

---

<p align="center">
      Notes by <a href="https://github.com/MK2112">MK2112</a>
</p>
