# Visual Taxonomy - Predicting Attributes from Product Images

Deep learning model for automated product attribute prediction in e-commerce.

**Final Rank:** 🏅 **30th out of 2,352 teams**  
**Score:** 📊 **0.75331**

This project was developed for the **Visual Taxonomy Challenge** sponsored by Meesho on Kaggle (November 2024). The challenge focused on building an intelligent system that predicts multiple product attributes from fashion images, enhancing product listing accuracy and reducing manual entry errors in e-commerce platforms.

## 🎯 Problem Statement

Build a deep learning model to predict multiple product attributes (color, pattern, style, material, etc.) using only product images. The system must handle:
- **Multiple categories** (Sarees, Women Tops & Tunics, Women T-shirts, etc.)
- **Category-specific attributes** (up to 10 attributes per category)
- **Class imbalance** across different attribute values
- **Missing/invalid labels** in training data

## 🏗️ Architecture Overview

### Core Components

1. **Enhanced ResNet50 Backbone**
   - Pre-trained ResNet50 with selective layer unfreezing
   - Squeeze-and-Excitation (SE) blocks for enhanced feature extraction
   - Adaptive feature pooling for consistent representation

2. **Multi-Head Architecture**
   - Category-specific attribute heads
   - Dynamic routing based on product category
   - Complex vs. simple attribute handling

3. **Advanced Loss Function**
   - Weighted Focal Loss for class imbalance
   - Per-class weighting based on inverse frequency
   - Gradient clipping for stable training

4. **Data Pipeline**
   - Intelligent data augmentation
   - Weighted sampling for balanced training
   - Missing label handling with class weights

## 📊 Model Architecture

```
Input Image (224x224x3)
        ↓
   ResNet50 Backbone
   (layers 3-4 trainable)
        ↓
   SE Block (2048 channels)
        ↓
  Global Average Pooling
        ↓
   Category-Specific Heads
   ┌─────────────────────┐
   │ Simple Attributes   │ → Linear(2048→256→classes)
   │ Complex Attributes  │ → Linear(2048→512→256→classes)
   └─────────────────────┘
```

## 🚀 Key Features

- **Intelligent Class Weighting**: Automatic calculation of class weights using inverse frequency with smoothing
- **Category-Aware Training**: Different attribute heads for each product category
- **Advanced Augmentation**: Color jittering, rotation, affine transforms with controlled randomness
- **Robust Validation**: Comprehensive F1-score calculation with micro/macro averaging
- **Memory Efficient**: Optimized data loading with proper batch handling
- **Early Stopping**: Prevents overfitting with patience-based stopping

## 📋 Requirements

```bash
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install Pillow tqdm
pip install matplotlib seaborn  # for visualization
```

## 💻 Usage

### Training

```python
# Basic training
python main.py

# Custom parameters
python main.py --epochs 10 --batch_size 64 --lr 0.001
```

### Inference

```python
from main import ImprovedAttributeClassifier, predict
import torch
from torchvision import transforms

# Load model
model = ImprovedAttributeClassifier(dataset)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict single image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

predictions = predict(model, 'path/to/image.jpg', transform, device)
print(predictions)
```

## 📈 Performance Metrics

| Category | F1 Score | Attributes |
|----------|----------|------------|
| Sarees | 0.78 | Fabric, Color, Pattern, Border, etc. |
| Women Tops & Tunics | 0.74 | Sleeve, Neckline, Fit, Pattern, etc. |
| Women T-shirts | 0.72 | Sleeve, Fit, Neckline, Print, etc. |
| **Overall** | **0.753** | **Multi-category Average** |

## 🎛️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | AdamW optimizer |
| Batch Size | 32 | Memory-efficient training |
| Epochs | 5 | With early stopping |
| Weight Decay | 0.01 | L2 regularization |
| Focal Loss γ | 2.0 | Focus on hard examples |
| Dropout | 0.3-0.4 | Prevent overfitting |
| Image Size | 224×224 | Standard ResNet input |

## 🔧 Advanced Features

### Weighted Focal Loss
```python
class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma      # Focus on hard examples
        self.alpha = alpha      # Class-specific weights
```

### Squeeze-and-Excitation Block
```python
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        # Channel attention mechanism
        # Improves feature representation quality
```

### Category-Specific Heads
```python
# Different complexity based on attribute difficulty
if self.is_complex_attribute(category, attr_name):
    # 3-layer network for complex attributes
    head = nn.Sequential(Linear(2048→512), GroupNorm, ReLU, 
                        Linear(512→256), GroupNorm, ReLU, 
                        Linear(256→classes))
else:
    # 2-layer network for simple attributes
    head = nn.Sequential(Linear(2048→256), GroupNorm, ReLU,
                        Linear(256→classes))
```

## 📊 Results Analysis

### Strengths
- **High Accuracy**: Achieved top 1.3% ranking (30/2352)
- **Robust Handling**: Effective management of class imbalance
- **Category Awareness**: Specialized handling for different product types
- **Efficient Training**: Convergence in just 5 epochs

### Key Insights
- SE blocks provided 2-3% improvement in F1 scores
- Weighted Focal Loss crucial for handling rare attribute classes
- Category-specific architecture better than unified approach
- Proper data augmentation prevented overfitting

## 🚧 Future Improvements

- [ ] **Ensemble Methods**: Combine multiple model architectures
- [ ] **Advanced Augmentation**: CutMix, MixUp for better generalization
- [ ] **Attention Mechanisms**: Visual attention for attribute localization
- [ ] **Multi-Scale Features**: Feature Pyramid Networks for detail capture
- [ ] **Knowledge Distillation**: Transfer learning from larger models


## 🙏 Acknowledgements

- **Meesho** for sponsoring the competition
- **Kaggle** for hosting the platform
- **PyTorch Community** for excellent documentation
- **Competition Participants** for healthy competition and knowledge sharing

---

**Built with ❤️ for the Fashion AI Community**
