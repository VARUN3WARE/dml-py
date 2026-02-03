# Core Concepts

Understanding the fundamental concepts behind PyDML.

## Deep Mutual Learning

Deep Mutual Learning (DML) is a collaborative learning approach where multiple neural networks of the same task learn simultaneously from each other via mimicry.

### Key Principles

**Bidirectional Knowledge Transfer**
: Unlike traditional knowledge distillation (teacher→student), DML allows all networks to teach and learn from each other simultaneously.

**Peer Learning**
: Networks act as peers, with no fixed teacher-student hierarchy.

**Mimicry Loss**
: Networks minimize the KL divergence between their output distributions, encouraging consensus while maintaining diversity.

### The DML Loss Function

For a batch of data, each network $i$ optimizes:

$$
L_i = L_{CE}(y_i, y_{true}) + \sum_{j \neq i} L_{KL}(y_i, y_j)
$$

Where:

- $L_{CE}$ is the cross-entropy loss with ground truth
- $L_{KL}$ is the KL divergence loss between network outputs
- $y_i$ are the softmax outputs of network $i$
- $y_{true}$ are the ground truth labels

### Benefits

1. **Better Generalization**: Peer learning acts as implicit regularization
2. **Ensemble Performance**: Multiple trained models can be ensembled
3. **No Teacher Required**: Unlike distillation, no pre-trained teacher needed
4. **Scalable**: Works with any number of networks

## Knowledge Distillation

Knowledge distillation transfers knowledge from a large teacher model to a smaller student model.

### Temperature Scaling

Softening the output distributions with temperature $T$:

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Higher temperatures produce softer probability distributions that reveal inter-class relationships.

### Distillation Loss

$$
L = \alpha L_{CE}(y_{student}, y_{true}) + (1-\alpha) T^2 L_{KL}(y_{student}, y_{teacher})
$$

The $T^2$ factor ensures gradient magnitudes remain approximately constant.

## Feature-Based Learning

Instead of only matching output distributions, networks can learn from intermediate layer representations.

### Feature Matching

Minimize distance between feature maps:

$$
L_{feature} = ||f_i^{(l)} - f_j^{(l)}||^2
$$

Where $f^{(l)}$ represents features from layer $l$.

### Attention Transfer

Match attention maps (spatial importance):

$$
L_{attention} = ||A_i - A_j||^2
$$

Where $A = \sum_c |F_c|^p$ for feature map $F$ and typically $p=2$.

## Ensemble Learning

Combining predictions from multiple models for better performance.

### Ensemble Strategies

**Average Ensemble**
: Average the class probabilities

$$
p_{ensemble} = \frac{1}{N} \sum_{i=1}^N p_i
$$

**Weighted Ensemble**
: Weight by validation performance

$$
p_{ensemble} = \sum_{i=1}^N w_i p_i, \quad \sum w_i = 1
$$

**Voting Ensemble**
: Majority voting on predicted classes

## Training Dynamics

### Collaborative Training Loop

1. **Forward Pass**: All networks process the same batch
2. **Peer Loss**: Compute KL divergence between network outputs
3. **Combined Loss**: Supervised loss + peer learning losses
4. **Backward Pass**: Each network updates independently
5. **Synchronization**: Networks implicitly coordinate through shared batch

### Convergence Behavior

- Networks converge to similar but distinct solutions
- Diversity maintained through different initializations
- Ensemble benefits from complementary errors

## Advanced Concepts

### Peer Selection

Not all peers are equally helpful. Strategies include:

- **All Peers**: Learn from every other network
- **Best Peer**: Learn only from highest-performing peer
- **Dynamic Selection**: Adapt peer set during training
- **Random Sampling**: Randomly select peers each iteration

### Curriculum Learning

Gradually increase training difficulty:

- Start with easy examples
- Progress to harder examples
- Can be combined with peer selection

### Temperature Scheduling

Dynamically adjust temperature during training:

- Start high for exploration
- Decrease for exploitation
- Can improve final performance

## Next Steps

- See [Trainers](trainers.md) for implementation details
- Read [Models](models.md) for architecture considerations
- Check [Utilities](utilities.md) for helper functions
