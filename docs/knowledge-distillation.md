### Group 8: Stephen Ebrahim and Ebram Mekhail

# Milestone 4: Model Compression

### Overview of Model Compression:

As it has been observed from the Gradient Boosting algorithm, using a plethora of models to construct a model proves to be very accurate - this is true for all ensemble techniques. The Ensemble Predictions technique involves the usage of multiple weak models that support each other and ultimately together form an accurate model. In other words, all of the models have a unique pattern that it has observed, and together these models combine their "knowledge" in a way to formulate one well-rounded versatile model which produces exceptional results. However, it is evident that the abundance of these models proves to be very costly in computation for regular consumers that do not have expensive and powerful machines. Subsequently, because this is a major issue, there have been numerous approaches in attempting to deal with it through some sort of model compression technique. The paper by Hinton et al. titled _Distilling the Knowledge in a Neural Network (2015)_ does a great job describing and addressing this issue.

Hinton et al. use a clever analogy of insects and their larvae to relate it to how their technique is developed and how it operates. When an insect is in its larva stage, the larval shell forms according to its surrounding environment and it is optimized in a way so that it strives in that specific environment. Essentially, in that stage, the insect will build its own environment (the shell) to account for the surrounding conditions. The technique that is developed in the paper uses a similar idea; the model will have two separate environments, one for development, and one for deployment. The deployment stage will extract the "knowledge" attained from the development stage without having to go through the expensive computations and training that is experienced there.

### Knowledge Distillation (KD):

Knowledge Distillation refers to the idea discussed above; the model that will be used in production is a simple model that attempts to get as much of the valuable information in the complicated trained model while remaining straightforward. The following figure is an exceptional example that illustrates the knowledge distillation idea in terms of Neural Networks.

<p align="center">
  <img src="./Teacher-Student-Distillation-Model.png" alt="Teacher-Student Distillation Model" style="width:70%;"/><br>
  <sub><sup>Knowledge Distillation: A Survey, Gou et al. (2020)
  https://arxiv.org/abs/2006.05525</sup></sub>
</p>

This figure is from a paper called _Knowledge Distillation: A Survey_ by Gou et al. (2020). Since the teacher has a substantial amount of knowledge, it is considered the more complicated training model. The student, on the other hand, is the simpler model that is attempting to retain as much information as possible without making it too complicated for them to understand.

A central idea in Distillation is the use of "soft-max" which in short, is a function that takes in a logit and produces a probability for each of the class presents. The following equation from the paper by Hinton et al. depicts this idea.
$$q_i-\frac{exp(z_i/T)}{\sum_jexp(z_j/T)}$$
where $q_i$ represents the probability, $z_i, z_j$ represents the logits, and $T$ is the "temperature" which is normally set to 1 and is used to control the softness of the probability distribution.

Essentially, soft-max will take in a vector in the training process and produce a vector of probabilities. This vector of probabilities represents the impact imposed by that vector on the model and training process. Assigning a value to this impact (the probability) gives a good estimate of whether or not this specific part of the model should be factored into the model that will be used for production. Another way to think about it is that if the probability is high, then that specific part of the model should be distilled and then transferred into the simpler model.

The temperature is an important factor in distilling the model because it helps with how the model picks up information in a sense. As mentioned earlier, the temperature $T$ helps with "softening" in the soft-max functions. The reason this is important is that it helps the model choose what to distill easier. In a sense, if there are a lot of high probability values returned from the soft-max function, the model obviously cannot pick all of them as it will consequently create another complicated value. Instead, the temperature helps minimize these probabilities and make them "softer" which helps the model choose what to distill easier. Through this technique, and varying the value of the temperature $T$ variable, knowledge distillation can be achieved effectively in many scenarios.

### References:

[1] Distilling the Knowledge in a Neural Network (Mar 2015)
Geoffrey Hinton, Oriol Vinyals, Jeff Dean
https://arxiv.org/abs/1503.02531

[2] Knowledge Distillation: A Survey, Gou et al. (May 2020)
https://arxiv.org/abs/2006.05525

<br>
<hr>
<br>

## Part 2 (Cont'd): Results
