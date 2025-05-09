# Custom Loss Function Design for Nuanced Sentiment Analysis

This document details the design choices, theoretical justifications, and inherent tradeoffs of the custom loss functions implemented in `train_utils.py`: `SentimentWeightedLoss` and the more comprehensive `SentimentFocalLoss`.

## Core Objective: Enhancing Sentiment Classification

We aim to:
1.  Improve model calibration (i.e., ensure prediction confidence aligns with actual correctness).
2.  Focus learning on more informative or challenging samples.
3.  Regularize the model to prevent overconfidence and enhance generalization.
4.  Incorporate heuristics relevant to textual data, such as review length.

Both implemented loss functions build on the standard BCE loss, which computes the binary cross-entropy between the raw logits output by the model and the true binary labels, providing a per-sample loss. This serves as the `base_loss`.

## `SentimentWeightedLoss`: Heuristic-Driven Sample Weighting

The `SentimentWeightedLoss` class applies a per-sample weighting scheme to the base BCE loss. This weighting is a product of two heuristic components: length-based weighting and confidence-based weighting. The combined weights are then normalized across the batch.

### **Length Weighting (`length_weight`)**

This component hypothesizes that the length of a review may correlate with its information content or the effort invested by the reviewer, potentially indicating a more reliable or nuanced sentiment signal. Very short reviews might be ambiguous or less considered.

The weight is calculated as $w_{\text{length}} = \frac{\sqrt{L}}{\sqrt{L_{\text{max\_batch}}}}$, where $L$ is the number of tokens in the current review and $L_{\text{max\_batch}}$ is the maximum number of tokens in any review within the current batch. 

The square root function ($L \mapsto \sqrt{L}$) is chosen to provide diminishing returns; the importance increases with length but at a decreasing rate, preventing exceptionally long reviews from disproportionately dominating the loss. The implementation further clamps this weight: `length_weight.clamp(self.min_len_weight_sqrt, 1.0)`, ensuring a minimum influence even for very short reviews and capping the maximum weight at 1. The `min_len_weight_sqrt` is initialized to 0.1.

**Impact on Training Dynamics & Model Behavior:** The model will assign higher importance (and thus larger gradient updates) to errors made on longer reviews within a batch. This may encourage the model to learn features that are more prevalent or clearly expressed in more extensive texts. Haven't had a chance to test this yet though.

**Tradeoffs & Potential Issues:**
* **Assumption Validity:** The core assumption that length correlates positively with signal quality may not universally hold. Longer reviews can also contain more noise, off-topic content, or diluted sentiment. It is also possible that the prevalence of phrases with conflicting sentiment might be higher in longer reviews.
* **Batch Dependency:** Normalization by $L_{\text{max\_batch}}$ makes the absolute weight of a fixed-length review dependent on the composition of its current batch, potentially introducing minor training variance.
* **Model Bias:** The model might inadvertently learn to prioritize verbosity over concise, potent sentiment expressions.

### **Confidence Weighting (`confidence_weight`)**

The idea here is to modulate the loss based on the model's own output confidence and make the model pay more attention to the predictions that it makes with high certainty, heavily penalizing confident mistakes and reinforcing confident correct predictions.

The weight is $w_{\text{conf}} = |\sigma(\text{logit}) - 0.5| \times 2$. Here, $\sigma(\text{logit})$ is the predicted probability $p$.
* If $p \approx 0.5$ (low confidence), $w_{\text{conf}} \approx 0$.
**Impact on Training Dynamics & Model Behavior:**
* **Confident Wrong Predictions:** If the model is confident ($p \approx 0$ for target 1, or $p \approx 1$ for target 0), $w_{\text{conf}}$ is high. Since the `base_loss` (BCE) is already very large for such cases, the effective loss ($L_{\text{BCE}} \times w_{\text{conf}}$) is significantly amplified.

* **Confident Correct Predictions:** If the model is confident and correct ($p \approx 1$ for target 1), $w_{\text{conf}}$ is high. The `base_loss` is very low. The effective loss remains low but is scaled up relative to what it would be if it were an uncertain correct prediction. In other words, its kind of BCE on steroids, encouraging the model to be really decisive when correct.
* **Uncertain Predictions:** If $p \approx 0.5$, $w_{\text{conf}}$ is low, there is a dampening of the loss (and gradients) for these samples, irrespective of correctness.

**Tradeoffs & Potential Issues:**
* **Model Timidity:** The model might become overly cautious, with predicted probabilities clustering around 0.5 to avoid the amplified penalties from confident mistakes. Or if most of the samples for which the model is confident and correct, and they are the vast majority of samples in the training corpus, the model could become overconfident. 
* **Redundancy with BCE?** So why add confidence weighting when BCE already penalizes wrong predictions quite hard, especially confident ones?
    * So BCE does indeed penalize confident errors harshly due to its logarithmic nature (e.g., $-\log(p)$ term). That said, the confidence weight acts as an *additional, explicit modulator* based on the *degree* of confidence itself.
    * **Emphasis and Shaping:** While BCE's penalty increases as probabilities approach 0 when $y=1$, the confidence weight $w_{\text{conf}}$ provides a *further multiplicative emphasis* explicitly targeting how far this $p$ is from $0.5$. It says "not only were you wrong, but you were *very sure* you were right (or very sure about the wrong class), so this error is even more critical." It sculpts the loss landscape to create even steeper penalties for confident errors.
    * **Reinforcing Confident Correctness:** BCE yields very small losses for confident correct predictions. The $w_{\text{conf}}$ ensures that these small losses, for highly confident *correct* predictions, are still weighted more than the small losses for *uncertain correct* predictions. This subtly encourages the model to not just be correct, but to be confidently correct, potentially leading to more robust feature learning for clear-cut cases.
    * **Calibration Pressure:** By explicitly linking the loss magnitude to prediction extremity, this weight can exert additional pressure on the model to improve the calibration of its confidence scores.
    * Thus, $w_{\text{conf}}$ isn't redundant but rather serves as a mechanism to further refine the model's attention and penalty structure based on its output confidence profile, above and beyond the inherent properties of BCE.

### **Combined Weight Normalization**

The individual length and confidence weights are multiplied ($w = w_{\text{length}} \times w_{\text{conf}}$). This combined weight $w$ is then normalized by its batch mean: $w_{\text{norm}} = w / (\text{mean}(w) + \epsilon)$.

Normalizing the weights to have an approximate mean of 1 across the batch is crucial. It ensures that the custom weighting scheme primarily *redistributes* the importance of samples within the batch, rather than systematically increasing or decreasing the overall magnitude of the loss. This stabilizes training by preventing the effective learning rate from being implicitly altered by the weighting scheme.

## `SentimentFocalLoss`: Adding Label Smoothing and Focal Loss

The `SentimentFocalLoss` class builds upon the heuristic weighting principles of `SentimentWeightedLoss` and additionally integrates Label Smoothing and Focal Loss. It aims for an even more robust and nuanced training process. The default values are `gamma_focal=0.1` and `label_smoothing_epsilon=0.05`. In both cases, the default values are chosen to be very conservative. 

In this particular loss function, we're throwing a lot of stuff at the wall to see what sticks. Given more time, it would make sense to do a more systematic investigation of the effects of different components of the loss function.

### **Label Smoothing (`label_smoothing_epsilon`)**

Standard classification models trained with hard labels (0 or 1) can become overconfident and may not generalize well, as they are encouraged to produce extreme logits. Label smoothing introduces a form of regularization by preventing the model from becoming too certain about its predictions.

With the smoothed label the BCE loss becomes

$$
\mathcal{L}_{\text{BCE}}
=-\Bigl[
y_{\text{true(smooth)}}\;\log\bigl(y_{\text{pred}}\bigr)
\;+\;
\bigl(1-y_{\text{true(smooth)}}\bigr)\;\log\bigl(1-y_{\text{pred}}\bigr)
\Bigr].
$$

Instead of training against hard targets of either 0 or 1 - i.e. $y \in \{0, 1\}$, smoothed targets $y'_{\text{ls}}$ are used. If the original target $y$ is 1, $y'_{\text{ls}} = 1 - \epsilon_{\text{ls}}$. If $y$ is 0, $y'_{\text{ls}} = \epsilon_{\text{ls}}$. (The implementation uses $y'_{\text{ls}} = y \cdot (1-\epsilon_{\text{ls}}) + (1-y) \cdot \epsilon_{\text{ls}}$). The BCE loss is then computed using these $y'_{\text{ls}}$. This means the model is penalized if it predicts exactly 1 or 0 for the respective classes, as the "ideal" smoothed target is slightly inset from these extremes.

**Numerical example — positive class**

| variable               | value                           |
| ---------------------- | ------------------------------- |
| hard label             | $y_{\text{true}} = 1$           |
| smoothing parameter    | $\varepsilon = 0.1$             |
| smoothed label         | $y_{\text{true(smooth)}} = 0.9$ |
| model output (sigmoid) | $y_{\text{pred}} = 0.7$         |

**Impact on Training Dynamics & Model Behavior:**
* Reduces the magnitude of logits learned by the model.
* Encourages the differences between logits for the correct class and other classes to be a more constant quantity, rather than pushing one logit to infinity.
* Improves model calibration: the predicted probabilities tend to better reflect the true likelihood of correctness.
* Often leads to improved generalization and robustness, particularly for large models like BERT that can easily overfit to training set specifics.
* In our specific case, it kind of mitigates the effect of the confidence weighting, as the smoothed label is less extreme.

**Tradeoffs & Potential Issues:**
* **Slight Performance Decrease on Training Set:** The model will not achieve "perfect" scores on the training data in terms of matching the original hard labels.
* **Obscuring True Probabilities (if $\epsilon_{\text{ls}}$ too large):** If the smoothing factor is too large, it can make the learning task overly ambiguous and hinder the model's ability to discriminate effectively, leading to underfitting. The default of 0.05 is a common, mild value.

### **Focal Loss Modulation (`gamma_focal`)**

Focal Loss was originally proposed to address extreme foreground-background class imbalance in object detection. Its core idea of down-weighting the loss assigned to well-classified (easy) examples is broadly applicable for focusing the model's attention on hard-to-classify (hard) examples, which can sometimes help to capture nuances.

In Focal Loss, the standard BCE loss for a sample is multiplied by a modulating factor.
* **Standard Focal Loss (`gamma_focal > 0`): Modulator is $(1-p_t)^{\gamma_{\text{focal}}}$**. $p_t$ is the predicted probability for the *original ground truth class*.
* For easy, well-classified examples ($p_t \approx 1$), $(1-p_t)^{\gamma_{\text{focal}}}$ is very small, significantly reducing their loss contribution.
* For hard, misclassified examples ($p_t \approx 0$), $(1-p_t)^{\gamma_{\text{focal}}} \approx 1$, so their loss contribution (already high from BCE) is largely preserved or only slightly reduced.
* **Reversed Focal Loss (`gamma_focal < 0`): Modulator is $(p_t)^{|\gamma_{\text{focal}}|}$**.
    * This experimental variant up-weights easy examples ($p_t \approx 1 \implies \text{modulator} \approx 1$) relative to hard examples ($p_t \approx 0 \implies \text{modulator} \approx 0$), effectively making the model focus more on perfecting its predictions for samples it already finds easy.
* If `gamma_focal = 0`, the modulator is 1, and no focal effect is applied. The default of `gamma_focal=0.1` suggests a very gentle application of the standard focal effect.

**Impact on Training Dynamics & Model Behavior:**
* **Standard FL:** The training process dynamically prioritizes difficult examples. The model is forced to allocate more capacity to learn the decision boundary in complex regions of the feature space. This can be particularly useful for sentiment analysis where reviews with sarcasm, subtlety, or mixed opinions are harder to classify.
* **Reversed FL:** The model would predominantly learn from easy examples. This is generally not desired for robust generalization but is included for experimental completeness. Haven't tested this yet.

**Tradeoffs & Potential Issues:**
* **Standard FL:**
    * *Slower Convergence:* If $\gamma_{\text{focal}}$ is large, down-weighting many examples can slow down learning initially.
    * *Overfitting to Outliers:* Extreme focus on very hard examples might lead to overfitting on noisy or atypical samples if $\gamma_{\text{focal}}$ is too high. The chosen default of 0.1 is very mild, mitigating this risk.
* **Reversed FL:** Likely to lead to poor generalization on complex examples.


### **Length and Confidence Weighting in `SentimentFocalLoss`**

* The `SentimentFocalLoss` retains the `confidence_w` and `length_w` calculations identical to those in `SentimentWeightedLoss`. These are computed and combined into `external_weights`.
* These `external_weights` are then normalized (as before) and are used to multiply the `modulated_loss_terms` (which are the `base_bce_loss_terms` already multiplied by the `focal_modulator`).

This means the heuristic weightings for length and confidence are applied *on top* of the focal modulation and label-smoothed BCE loss. The model first adjusts its focus based on example difficulty (Focal Loss) and target softness (Label Smoothing), and then further refines the per-sample loss based on review length and its own prediction confidence for that sample. This layered approach allows for multiple facets of sample importance and learning dynamics to be addressed. Like I said, this is a lot of stuff to throw at the wall, and really isn't very scientific, was kind of a YOLO run.
