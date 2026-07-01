# 2.4 多模态大模型 - 关键知识点详解

> 本节为 2.4 节的补充知识点，包含 CLIP/LLaVA 架构解析、多模态对齐技术、实战代码。

---

## 知识点 1: 多模态架构对比

### 主流多模态模型对比表

| 模型 | 架构 | 输入模态 | 输出模态 | 参数量 | 开源性 |
|------|------|---------|---------|-------|-------|
| **CLIP** | 双塔 ViT+Text | 图像 + 文本 | 嵌入向量 | 400M | ✅ |
| **ALIGN** | 双塔 EffNet+BERT | 图像 + 文本 | 嵌入向量 | - | ❌ |
| **Flamingo** | Perceiver Resampler + LLM | 图像 + 文本 | 文本 | 80B | ❌ |
| **BLIP-2** | Q-Former + LLM | 图像 + 文本 | 文本 | - | ✅ |
| **LLaVA** | 线性投影 + LLM | 图像 + 文本 | 文本 | 7B-13B | ✅ |
| **MiniGPT-4** | 线性投影 + Vicuna | 图像 + 文本 | 文本 | 13B | ✅ |

### 架构演变图

```
多模态架构演变

2021: CLIP (对比学习)
┌─────────┐    ┌─────────┐
│  Image  │    │  Text   │
│ Encoder │    │ Encoder │
│  (ViT)  │    │(Transformer)│
└────┬────┘    └────┬────┘
     │              │
     └──────┬───────┘
            │
     Contrastive Loss
     (拉近匹配对，推开非匹配对)

2022: ALIGN/ Florence (大规模预训练)
         同样的双塔架构 + 更大规模数据

2023: Flamingo/BLIP-2 (桥接模块)
┌─────────┐     ┌─────────┐
│  Image  │     │  Text   │
│ Encoder │     │ Encoder │
│  (ViT)  │     │         │
└────┬────┘     └────┬────┘
     │               │
     ▼               │
┌─────────┐          │
│ Perceiver │         │
│  Resampler│         │
│  (Q-Former)│        │
└────┬────┘          │
     │                │
     └──────┬─────────┘
            │
     ┌──────▼──────┐
     │   LLM Decoder│
     │  (生成文本)  │
     └─────────────┘

2023: LLaVA (简单有效)
┌─────────┐
│  Image  │
│ Encoder │
│ (ViT-L) │
└────┬────┘
     │
     ▼
┌─────────┐
│  Linear │  ← 简单的线性投影
│ Project │
└────┬────┘
     │
     ▼
┌─────────┐
│  LLM    │  ← Vicuna/Llama
│ (Decoder)│
└─────────┘
```

---

## 知识点 2: CLIP 详解

### CLIP 原理与代码实现

```python
# examples/2-4-multimodal/clip_impl.py
"""
CLIP (Contrastive Language-Image Pre-training) 实现

核心思想：
1. 图像和文本编码到同一嵌入空间
2. 对比学习：匹配的图文对拉近，不匹配的推开
3. 零样本迁移：通过文本提示进行分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel


class CLIPModel(nn.Module):
    """
    简化版 CLIP 模型
    
    组件:
    1. Image Encoder (ViT)
    2. Text Encoder (Transformer)
    3. 投影层 (将两者映射到同一空间)
    """
    
    def __init__(
        self,
        image_emb_dim: int = 768,
        text_emb_dim: int = 768,
        proj_dim: int = 512,
        temperature: float = 0.07
    ):
        super().__init__()
        
        # 图像编码器 (使用预训练 ViT)
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.image_encoder = vit
        self.image_encoder.heads = nn.Identity()  # 移除分类头
        
        # 文本编码器 (使用 BERT)
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # 投影层 (映射到同一空间)
        self.image_proj = nn.Sequential(
            nn.Linear(image_emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_emb_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / temperature)))
        
        # 初始化投影层
        self._init_proj()
    
    def _init_proj(self):
        """初始化投影层为单位映射"""
        nn.init.eye_(self.image_proj[0].weight)
        nn.init.zeros_(self.image_proj[0].bias)
        nn.init.eye_(self.text_proj[0].weight)
        nn.init.zeros_(self.text_proj[0].bias)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images: (batch, 3, 224, 224)
        
        Returns:
            image_features: (batch, proj_dim), L2 归一化
        """
        # ViT 编码
        image_emb = self.image_encoder(images)  # (batch, image_emb_dim)
        
        # 投影 + L2 归一化
        image_proj = self.image_proj(image_emb)
        image_proj = F.normalize(image_proj, dim=-1)
        
        return image_proj
    
    def encode_text(self, texts: list) -> torch.Tensor:
        """
        编码文本
        
        Args:
            texts: list of strings
        
        Returns:
            text_features: (batch, proj_dim), L2 归一化
        """
        # Tokenize
        tokens = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        ).to(next(self.parameters()).device)
        
        # BERT 编码 (使用 [CLS] token)
        outputs = self.text_encoder(**tokens)
        text_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # 投影 + L2 归一化
        text_proj = self.text_proj(text_emb)
        text_proj = F.normalize(text_proj, dim=-1)
        
        return text_proj
    
    def forward(
        self,
        images: torch.Tensor,
        texts: list
    ) -> dict:
        """
        前向传播 (训练用)
        
        返回:
            image_features: 图像嵌入
            text_features: 文本嵌入
            logits: 相似度矩阵
            labels: 对比学习标签
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        # 计算相似度矩阵
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        
        # 对比学习标签 (对角线是正样本)
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=images.device)
        
        return {
            'image_features': image_features,
            'text_features': text_features,
            'logits': logits,
            'labels': labels
        }
    
    def compute_loss(self, output: dict) -> torch.Tensor:
        """
        计算对比损失
        
        CLIP 使用对称交叉熵损失:
        - 图像→文本 方向
        - 文本→图像 方向
        """
        logits = output['logits']
        labels = output['labels']
        
        # 对称交叉熵损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


# ==================== 使用示例 ====================

class CLIPEvaluator:
    """
    CLIP 评估器
    
    用途:
    1. 零样本图像分类
    2. 图文检索
    3. 图像描述
    """
    
    def __init__(self, model: CLIPModel, device: str = 'cuda'):
        self.model = model.to(device).eval()
        self.device = device
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def classify(self, image, class_names: list) -> str:
        """
        零样本图像分类
        
        Args:
            image: PIL Image
            class_names: 类别名称列表
        
        Returns:
            预测的类别
        """
        # 处理图像
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # 编码图像
        image_feat = self.model.encode_image(image)
        
        # 编码类别提示
        prompts = [f"a photo of a {c}" for c in class_names]
        text_feats = self.model.encode_text(prompts)
        
        # 计算相似度
        similarity = image_feat @ text_feats.T  # (1, num_classes)
        probs = F.softmax(similarity * 100, dim=-1)
        
        # 返回最可能的类别
        pred_idx = probs.argmax().item()
        return class_names[pred_idx], probs[0, pred_idx].item()
    
    @torch.no_grad()
    def retrieve_images(
        self,
        query_text: str,
        image_candidates: list,
        top_k: int = 5
    ) -> list:
        """
        图文检索
        
        Args:
            query_text: 查询文本
            image_candidates: 候选图像列表
            top_k: 返回数量
        
        Returns:
            最相关的 top-k 图像索引
        """
        # 编码文本
        text_feat = self.model.encode_text([query_text])
        
        # 编码所有候选图像
        image_feats = []
        for img in image_candidates:
            img = self.transform(img).unsqueeze(0).to(self.device)
            img_feat = self.model.encode_image(img)
            image_feats.append(img_feat)
        
        image_feats = torch.cat(image_feats, dim=0)
        
        # 计算相似度
        similarities = text_feat @ image_feats.T  # (1, num_images)
        scores = similarities[0]
        
        # 返回 top-k
        top_indices = scores.topk(top_k).indices.tolist()
        return top_indices


# ==================== 零样本分类示例 ====================

if __name__ == "__main__":
    from PIL import Image
    
    # 加载模型
    model = CLIPModel()
    # model.load_state_dict(torch.load('clip_weights.pth'))
    
    evaluator = CLIPEvaluator(model)
    
    # 零样本分类
    image = Image.open('test_image.jpg').convert('RGB')
    class_names = ['cat', 'dog', 'bird', 'car', 'person']
    
    pred_class, confidence = evaluator.classify(image, class_names)
    print(f"预测：{pred_class} (置信度：{confidence:.2%})")
    
    # 图文检索
    query = "a dog playing in the park"
    candidate_images = [Image.open(f'image_{i}.jpg') for i in range(100)]
    top_indices = evaluator.retrieve_images(query, candidate_images, top_k=5)
    
    print(f"最相关的图像索引：{top_indices}")
```

---

## 知识点 3: LLaVA 架构详解

### LLaVA 信号处理流程

```
LLaVA 架构 (Llama-2 + ViT-L + Linear Projection)

输入图像 ──────┐
               ▼
        ┌──────────────┐
        │ ViT-L/16     │  ← 图像编码器 (冻结)
        │ (CLIP 预训练) │
        └──────┬───────┘
               │
               │ 图像特征 (24×24×1024)
               ▼
        ┌──────────────┐
        │ Linear Proj  │  ← 可训练投影层
        │ (2x MLP)     │
        └──────┬───────┘
               │
               │ 视觉嵌入 (24×24×4096)
               │
输入文本 ──────┼─────────────────────────┐
               ▼                         │
        ┌──────────────┐                 │
        │   Llama-2    │  ← LLM Decoder  │
        │   (7B/13B)   │                 │
        └──────┬───────┘                 │
               │                         │
               └─────────────────────────┘
                        │
                        ▼
                  输出文本
                    
关键设计:
1. ViT 和 LLM 都保持冻结 (QLoRA 微调)
2. 只训练投影层 (简单有效)
3. 多模态指令微调数据训练
```

### LLaVA 微调代码

```python
# examples/2-4-multimodal/llava_finetune.py
"""
LLaVA 风格多模态模型微调

基于预训练 CLIP + Llama，微调投影层实现 VQA 能力
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, ViTModel
from PIL import Image
from typing import Dict, List
import json


class LLaVAModel(nn.Module):
    """
    LLaVA 风格模型
    
    架构:
    - Image Encoder: CLIP ViT-L/14 (冻结)
    - Projector: 2 层 MLP (可训练)
    - LLM: Llama-2-7B (冻结或 LoRA)
    """
    
    def __init__(
        self,
        llm_name: str = "microsoft/Phi-3-mini-4k-instruct",
        vision_name: str = "openai/clip-vit-large-patch14",
        freeze_llm: bool = True
    ):
        super().__init__()
        
        # 1. 加载视觉编码器
        self.vision_model = ViTModel.from_pretrained(vision_name)
        
        # 锁定视觉编码器参数
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        vision_hidden_size = self.vision_model.config.hidden_size  # 1024
        
        # 2. 加载 LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            trust_remote_code=True
        )
        
        # 添加图像 token 到词表
        self.llm_tokenizer.add_special_tokens({
            'additional_special_tokens': ['<image>', '</image>']
        })
        
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # 扩展 embedding 层以包含新 token
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        if freeze_llm:
            for param in self.llm_model.parameters():
                param.requires_grad = False
        
        # 3. 视觉 - 语言投影器 (2 层 MLP)
        projector_hidden_dim = 2048
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden_size, projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, self.llm_model.config.hidden_size)
        )
        
        # 图像 token ID
        self.image_token_id = self.llm_tokenizer.convert_tokens_to_ids('<image>')
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像并投影到 LLM 空间
        
        Args:
            images: (batch, 3, 336, 336)
        
        Returns:
            image_embeds: (batch, num_patches, llm_dim)
        """
        with torch.no_grad():
            vision_outputs = self.vision_model(images, output_hidden_states=True)
        
        # 取最后一层输出
        image_features = vision_outputs.last_hidden_state  # (batch, num_patches, 1024)
        
        # 投影到 LLM 维度
        image_embeds = self.vision_proj(image_features)  # (batch, num_patches, llm_dim)
        
        return image_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            images: (batch, 3, 336, 336)
            labels: (batch, seq_len), -100 表示忽略
        """
        batch_size = input_ids.size(0)
        
        # 1. 编码图像
        image_embeds = self.encode_image(images)  # (batch, num_patches, llm_dim)
        
        # 2. 查找<image>token 位置
        image_positions = (input_ids == self.image_token_id)  # (batch, seq_len)
        
        # 3. 构建完整嵌入
        # 获取 LLM 的输入嵌入
        input_embeds = self.llm_model.get_input_embeddings()(input_ids)  # (batch, seq_len, llm_dim)
        
        # 将图像嵌入插入到<image>位置
        for i in range(batch_size):
            num_patches = image_embeds[i].size(0)
            num_images = image_positions[i].sum().item()
            
            if num_images > 0:
                # 找到第一个<image>位置
                img_pos = image_positions[i].nonzero()[0].item()
                
                # 替换该位置的嵌入为图像嵌入
                # 简化处理：用第一个图像 patch 替换
                # 实际 LLaVA 会将图像展开为序列
                input_embeds[i, img_pos] = image_embeds[i].mean(dim=0)
        
        # 4. 通过 LLM
        outputs = self.llm_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs.loss


# ==================== 数据集 ====================

class LVISDataset(Dataset):
    """
    简化版视觉语言数据集
    
    格式:
    [
        {
            "image": "path/to/image.jpg",
            "conversations": [
                {"role": "user", "content": "这张图片里有什么？"},
                {"role": "assistant", "content": "图片里有一只猫在沙发上。"}
            ]
        }
    ]
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_size: int = 336,
        max_length: int = 512
    ):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        
        # 加载图像
        image = Image.open(item['image']).convert('RGB')
        image = self.transform(image)
        
        # 构建对话
        conversations = item['conversations']
        
        # 格式化输入
        prompt = ""
        for conv in conversations:
            if conv['role'] == 'user':
                prompt += f"User: {conv['content']} "
            else:
                prompt += f"Assistant: {conv['content']} "
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 构建 labels (用于计算 loss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'images': image
        }


# ==================== 训练脚本 ====================

def train_llava():
    """训练 LLaVA 模型"""
    
    # 配置
    config = {
        'llm_name': 'microsoft/Phi-3-mini-4k-instruct',
        'vision_name': 'openai/clip-vit-large-patch14',
        'batch_size': 4,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'max_length': 512,
        'image_size': 336
    }
    
    # 初始化模型
    model = LLaVAModel(
        llm_name=config['llm_name'],
        freeze_llm=True  # 只训练投影层
    )
    model = model.cuda()
    
    # 优化器 (只优化投影层)
    optimizer = torch.optim.AdamW(
        model.vision_proj.parameters(),
        lr=config['learning_rate']
    )
    
    # 数据
    train_dataset = LVISDataset(
        data_path='data/llava_instruct.json',
        tokenizer=model.llm_tokenizer,
        max_length=config['max_length'],
        image_size=config['image_size']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # 训练循环
    model.train()
    
    for epoch in range(config['num_epochs']):
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            images = batch['images'].cuda()
            
            # 前向传播
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                labels=labels
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")
    
    # 保存模型
    model.save_pretrained('checkpoints/llava_finetuned')


if __name__ == "__main__":
    train_llava()
```

---

## 练习题

### 练习 1: 实现图文检索系统

使用 CLIP 实现一个简单的图文检索系统：
1. 准备 100 张图片的数据集
2. 编码所有图片的 CLIP 特征
3. 输入文本查询，返回最相关的 5 张图片

### 练习 2: 多模态数据增强

为以下场景设计多模态数据增强策略：
1. 医学影像报告生成
2. 商品图片描述
3. 街景图像问答

---

## 延伸阅读

- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [LLaVA 论文](https://arxiv.org/abs/2304.08485)
- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)

---

[← 返回 2.4 主文档](2-4-multimodal.md) | [下一章：第 3 章返回目录 →](../chapter-3/README.md)
