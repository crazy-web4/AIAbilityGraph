# 2.4 多模态大模型

> 多模态大模型突破文本边界，实现视觉 - 语言的深度融合。本节深入解析 CLIP、LLaVA 等代表性模型。

## 学习目标

学完本节后，你将能够：

- [ ] 理解多模态对比学习（CLIP 风格）的原理
- [ ] 掌握视觉 - 语言大模型（LLaVA 风格）的架构设计
- [ ] 实现图像描述、视觉问答等多模态任务
- [ ] 了解多模态训练的技巧与挑战

---

## 2.4.1 CLIP: 对比语言 - 图像预训练

### 核心思想

将图像和文本映射到同一向量空间，通过对比学习学习对齐表示。

```
                    CLIP 架构
    
    Image Encoder              Text Encoder
         │                           │
         │      对比学习              │
         │      ┌──────┐            │
    [图像特征] ──→│ Dot  │←── [文本特征]
         │        │Product│         │
         │        └──────┘         │
         │             │            │
         └─────────────┼────────────┘
                       │
              最大化正确配对相似度
              最小化错误配对相似度
```

### 损失函数

$$\mathcal{L}_{CLIP} = \frac{1}{2}\left(\mathcal{L}_{CE}(y, \hat{y}_{i2t}) + \mathcal{L}_{CE}(y, \hat{y}_{t2i})\right)$$

其中相似度计算：
$$\text{similarity}(\mathbf{I}, \mathbf{T}) = \frac{\mathbf{I} \cdot \mathbf{T}^T}{\|\mathbf{I}\|\|\mathbf{T}\|} \times \exp(\tau)$$

```python
# examples/2-4-multimodal/clip_from_scratch.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPModel(nn.Module):
    """
    CLIP 风格模型简化实现
    
    组件:
    - Image Encoder: Vision Transformer 或 ResNet
    - Text Encoder: Transformer Encoder
    - 对比学习头：温度缩放点积
    """
    
    def __init__(self, 
                 vocab_size=49408,
                 embed_dim=512,
                 image_size=224,
                 num_heads=8,
                 num_layers=12):
        super().__init__()
        
        # 图像编码器（简化版 ViT）
        self.image_encoder = VisionTransformer(
            image_size=image_size,
            patch_size=16,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # 文本编码器
        self.text_encoder = TransformerTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # 投影头（对齐到共享空间）
        self.image_projection = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.text_projection = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)
    
    def encode_image(self, images):
        """编码图像"""
        image_features = self.image_encoder(images)
        # L2 归一化
        image_features = F.normalize(image_features @ self.image_projection, dim=-1)
        return image_features
    
    def encode_text(self, text):
        """编码文本"""
        text_features = self.text_encoder(text)
        # 取 EOS token 作为句子表示
        text_features = text_features[torch.arange(text_features.shape[0]), text.argmax(dim=-1)]
        text_features = F.normalize(text_features @ self.text_projection, dim=-1)
        return text_features
    
    def forward(self, images=None, text=None):
        """
        训练模式：计算对比损失
        推理模式：返回特征
        """
        image_features = self.encode_image(images) if images is not None else None
        text_features = self.encode_text(text) if text is not None else None
        
        if image_features is not None and text_features is not None:
            # 计算 logits（图像到文本）
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T
            
            return logits_per_image, logits_per_text
        
        return image_features, text_features
    
    def compute_loss(self, logits_per_image, logits_per_text):
        """
        CLIP 对比损失
        """
        batch_size = logits_per_image.shape[0]
        
        # 标签：对角线是正样本
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # 双向交叉熵
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2


class VisionTransformer(nn.Module):
    """
    简易 ViT 实现
    """
    
    def __init__(self, image_size=224, patch_size=16, embed_dim=512, 
                 num_heads=8, num_layers=12):
        super().__init__()
        
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer 层
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # Patch 嵌入
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 位置编码
        x = x + self.pos_embed
        
        # Transformer
        for layer in self.transformer:
            x = layer(x)
        
        # 返回 CLS token
        return x[:, 0, :]


class TransformerTextEncoder(nn.Module):
    """
    简易 Transformer 文本编码器
    """
    
    def __init__(self, vocab_size, embed_dim, num_heads=8, num_layers=12):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 77, embed_dim))
        
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.token_embedding(x) + self.pos_embed[:, :x.shape[1], :]
        
        for layer in self.transformer:
            x = layer(x)
        
        return self.layer_norm(x)
```

---

## 2.4.2 LLaVA: 大型视觉 - 语言助手

### 架构设计

```
                     LLaVA 架构
    
        图像                    文本输入
         │                        │
         ↓                        ↓
    ┌─────────┐            ┌──────────┐
    │  ViT    │            │ LLM      │
    │ CLIP-L  │            │ (Llama)  │
    └────┬────┘            └────┬─────┘
         │                      │
         │ 图像特征             │ 文本嵌入
         │ (576×1024)           │
         ↓                      │
    ┌─────────┐                 │
    │ 投影层  │                 │
    │ (MLP)   │                 │
    └────┬────┘                 │
         │                      │
         │ 视觉 token            │
         │ (576×4096)           │
         └──────────┬───────────┘
                    │
                    ↓
              拼接输入序列
                    │
                    ↓
            ┌───────────────┐
            │  LLM 生成答案  │
            └───────────────┘
```

### 核心代码实现

```python
# examples/2-4-multimodal/llava_style_model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel


class LLaVAModel(nn.Module):
    """
    LLaVA 风格多模态模型
    
    组件:
    - 视觉编码器：CLIP ViT
    - 投影层：2 层 MLP
    - 语言模型：Llama / Vicuna
    """
    
    def __init__(self, 
                 llm_model_name="lmsys/vicuna-7b-v1.5",
                 vision_model_name="openai/clip-vit-large-patch14",
                 hidden_size=4096,
                 vision_hidden_size=1024):
        super().__init__()
        
        # 加载视觉编码器（冻结）
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_encoder.requires_grad_(False)
        
        # 加载语言模型
        self.language_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        # 多模态投影层（ViT 输出 → LLM 隐藏空间）
        self.multi_modal_projector = nn.Sequential(
            nn.Linear(vision_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 特殊 token
        self.image_token = "<image>"
        self.image_token_id = self.tokenizer.vocab_size
        self.tokenizer.add_tokens([self.image_token])
        self.language_model.resize_token_embeddings(len(self.tokenizer))
    
    def encode_image(self, images):
        """
        编码图像到视觉特征
        
        参数:
            images: (batch, 3, 224, 224)
        返回:
            image_features: (batch, num_patches, hidden_size)
        """
        with torch.no_grad():
            # ViT 输出: (batch, num_patches+1, vision_hidden_size)
            vision_outputs = self.vision_encoder(images, output_hidden_states=True)
            
            # 取最后一层 hidden state（去掉 CLS token）
            image_features = vision_outputs.hidden_states[-1][:, 1:, :]
        
        # 投影到 LLM 空间
        image_features = self.multi_modal_projector(image_features)
        
        return image_features
    
    def prepare_inputs(self, images, input_ids, attention_mask=None):
        """
        准备多模态输入
        
        将图像特征插入到文本序列中 <image> 位置
        """
        batch_size, seq_len = input_ids.shape
        
        # 编码图像
        image_features = self.encode_image(images)  # (batch, num_patches, hidden_size)
        
        # 获取文本嵌入
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 找到 <image> token 位置
        image_positions = (input_ids == self.image_token_id)
        
        # 替换图像位置的嵌入
        # 这里简化处理：假设每张图一个 <image> token
        
        # 创建新的输入嵌入，在图像位置插入视觉特征
        combined_embeds = text_embeds.clone()
        
        for i in range(batch_size):
            # 找到这张图的图像特征
            img_feat = image_features[i]  # (num_patches, hidden_size)
            
            # 找到 <image> 位置
            img_pos = image_positions[i].nonzero()
            if len(img_pos) > 0:
                img_idx = img_pos[0].item()
                # 将图像特征插入（这里简化为替换）
                # 实际 LLaVA 使用更复杂的拼接策略
                combined_embeds[i, img_idx:img_idx+1, :] = img_feat.mean(dim=0, keepdim=True)
        
        # 创建对应的 attention mask
        combined_attention_mask = torch.ones(
            batch_size, seq_len, device=input_ids.device
        )
        
        return combined_embeds, combined_attention_mask
    
    def forward(self, images, input_ids, labels=None, attention_mask=None):
        """
        前向传播
        
        参数:
            images: 图像 batch
            input_ids: 文本输入
            labels: 训练时的标签（用于计算损失）
        """
        # 准备多模态输入
        inputs_embeds, combined_attention_mask = self.prepare_inputs(
            images, input_ids, attention_mask
        )
        
        # 语言模型前向
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels,
        )
        
        return outputs
    
    def generate(self, images, input_ids, max_new_tokens=100):
        """
        生成式推理
        """
        self.eval()
        
        # 准备输入
        inputs_embeds, attention_mask = self.prepare_inputs(images, input_ids)
        
        # 使用语言模型的 generate
        generated = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        # 解码输出
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        return response
```

---

## 2.4.3 多模态训练

### 两阶段训练策略

#### 阶段 1: 预训练（特征对齐）

**目标**：训练投影层，使视觉特征对齐语言空间

**数据**：图像 - 文本对（如 LAION-400M）

**训练目标**：预测下一个文本 token，图像特征作为条件

```python
# examples/2-4-multimodal/train_multimodal.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
from PIL import Image
import json


class MultimodalDataset(Dataset):
    """
    多模态指令微调数据集
    
    格式:
    [
        {
            "image": "path/to/image.jpg",
            "text": "Describe this image: <image> The image shows..."
        }
    ]
    """
    
    def __init__(self, data_path, tokenizer, image_processor, max_length=512):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image = Image.open(item['image']).convert('RGB')
        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        
        # 处理文本
        text = item['text']
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'pixel_values': image,
            'input_ids': tokens['input_ids'][0],
            'attention_mask': tokens['attention_mask'][0],
            'labels': tokens['input_ids'][0].clone()
        }


def create_training_script():
    """
    创建完整的多模态训练脚本
    """
    
    training_args = TrainingArguments(
        output_dir="./output/multimodal-llava",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,           # 投影层学习率
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        fp16=True,
        gradient_checkpointing=True,
        
        # 优化器配置
        adam_beta2=0.999,
        max_grad_norm=1.0,
        
        # 冻结视觉编码器，只训练投影层 + LLM
    )
    
    # 训练策略:
    # 1. 只训练投影层（1epoch）
    # 2. 解冻 LLM，小学习率微调（4 epochs）
    
    return training_args


def train_2_stage(model, train_dataset):
    """
    两阶段训练策略
    
    阶段 1: 只训练投影层（特征对齐）
    阶段 2: 解冻 LLM，全模型微调
    """
    
    # ========== 阶段 1: 投影层预训练 ==========
    print("=" * 50)
    print("阶段 1: 投影层预训练")
    print("=" * 50)
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 只解冻投影层
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True
    
    training_args_1 = TrainingArguments(
        output_dir="./output/stage1",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=1e-3,  # 较高学习率
        warmup_steps=50,
        fp16=True,
    )
    
    trainer_1 = Trainer(
        model=model,
        args=training_args_1,
        train_dataset=train_dataset,
    )
    
    trainer_1.train()
    
    # ========== 阶段 2: 全模型微调 ==========
    print("\n" + "=" * 50)
    print("阶段 2: 全模型微调")
    print("=" * 50)
    
    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True
    
    # 视觉编码器可以用更小的学习率
    for param in model.vision_encoder.parameters():
        param.requires_grad = False  # 或者设置很小的 lr
    
    training_args_2 = TrainingArguments(
        output_dir="./output/stage2",
        num_train_epochs=4,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        warmup_steps=100,
        fp16=True,
        gradient_checkpointing=True,
    )
    
    trainer_2 = Trainer(
        model=model,
        args=training_args_2,
        train_dataset=train_dataset,
    )
    
    trainer_2.train()
    
    # 保存模型
    model.save_pretrained("./output/final-model")
```

---

## 2.4.4 多模态任务

### 1. 图像描述生成 (Image Captioning)

```python
# examples/2-4-multimodal/tasks/image_captioning.py

def image_captioning_demo(model, tokenizer, image_path):
    """
    图像描述生成
    
    Prompt: "Describe this image in detail: <image>"
    """
    from PIL import Image
    import torchvision.transforms as T
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # 构建 prompt
    prompt = "Describe this image in detail: <image>"
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # 生成描述
    with torch.no_grad():
        output = model.generate(
            image_tensor,
            inputs['input_ids'],
            max_new_tokens=100
        )
    
    print(f"图像描述：{output}")
    return output
```

### 2. 视觉问答 (VQA)

```python
# examples/2-4-multimodal/tasks/vqa.py

def vqa_demo(model, tokenizer, image_path, question):
    """
    视觉问答
    
    Prompt: "Question: What is in this image? Answer: <image>"
    """
    from PIL import Image
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 构建 prompt
    prompt = f"Question: {question}\nAnswer: <image>"
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # 生成答案
    answer = model.generate(
        image.unsqueeze(0),
        inputs['input_ids'],
        max_new_tokens=50,
        do_sample=False
    )
    
    print(f"Q: {question}")
    print(f"A: {answer}")
    return answer
```

### 3. 文档理解 (Document Understanding)

```python
# examples/2-4-multimodal/tasks/document_understanding.py

def document_qa_demo(model, tokenizer, document_image, question):
    """
    文档理解与问答
    
    适用：表格、图表、PDF 文档等
    """
    prompt = f"Based on the document below, answer the question.\n\nQuestion: {question}\n\n<Document> <image> </Document>\n\nAnswer:"
    
    # 高分辨率图像处理（文档需要）
    # LLaVA-1.5+ 支持多图/高分辨率
    
    # ... 实现省略
    pass
```

---

## 2.4.5 模型对比总结

| 模型 | 视觉编码器 | 语言模型 | 连接方式 | 参数量 |
|------|-----------|----------|----------|--------|
| **CLIP** | ViT-L/14 | Transformer | 对比学习 | 428M |
| **LLaVA-1.5** | CLIP ViT-L | Vicuna-7B | MLP 投影 | 7B |
| **BLIP-2** | ViT | Flan-T5 | Q-Former | 2-12B |
| **InstructBLIP** | ViT | Vicuna | Q-Former | 8-13B |
| **LLaVA-NeXT** | ViT | Llama-2 | 2×2 投影 | 7-34B |

---

## 练习题

### 基础题

1. **对比学习**：为什么 CLIP 要用双向对比损失（i2t + t2i）？

2. **投影层设计**：为什么 LLaVA 用 2 层 MLP 而不是线性投影？

3. **两阶段训练**：为什么不能直接端到端训练多模态模型？

### 编程题

4. 实现一个支持多图输入的 LLaVA 风格模型。

5. 在 COCO Caption 数据集上训练并评估图像描述生成性能。

---

## 延伸阅读

- 🌐 [CLIP Paper](https://arxiv.org/abs/2103.00020)
- 🌐 [LLaVA: Large Language and Vision Assistant](https://arxiv.org/abs/2304.08485)
- 🌐 [LLaVA-1.5 Technical Report](https://arxiv.org/abs/2310.03744)
- 🌐 [HuggingFace Multimodal Course](https://huggingface.co/learn/multimodal-course)

---

[← 上一节：2.3 参数高效微调](2-3-peft.md) | [第 3 章：大模型工程化 →](../chapter-3/README.md)
