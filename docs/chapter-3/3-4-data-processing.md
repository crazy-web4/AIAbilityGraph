# 3.4 训练数据处理

> 高质量数据是训练大模型的基础。本章讲解数据清洗、去重、质量评估等关键处理流程。

## 学习目标

学完本节后，你将能够：

- [ ] 构建大规模数据清洗流水线
- [ ] 实现文档级和语料级去重
- [ ] 设计数据质量评估指标
- [ ] 配置高效的数据加载器

---

## 3.4.1 数据清洗流水线

### 完整清洗流程

```python
# examples/3-4-data/data_cleaning.py
"""
大规模文本数据清洗流水线
"""

import re
import json
import hashlib
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class TextCleaner:
    """文本清洗器"""
    
    @staticmethod
    def remove_html(text: str) -> str:
        """移除 HTML 标签"""
        return re.sub(r'<[^>]+>', ' ', text)
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """移除 URL"""
        return re.sub(r'http[s]?://\S+', '', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """规范化空白字符"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def remove_special_chars(text: str) -> str:
        """移除特殊字符（保留基本标点）"""
        return re.sub(r'[^\w\s.,!?;:()\"\'-]', '', text)
    
    @staticmethod
    def fix_unicode(text: str) -> str:
        """修复 Unicode 编码问题"""
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    
    def clean(self, text: str) -> str:
        """完整清洗流程"""
        text = self.fix_unicode(text)
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        return text


class QualityFilter:
    """质量过滤器"""
    
    def __init__(self, 
                 min_length: int = 200,
                 min_words: int = 50,
                 max_word_repetition: float = 0.5,
                 min_stopword_ratio: float = 0.1,
                 max_symbol_ratio: float = 0.3):
        
        self.min_length = min_length
        self.min_words = min_words
        self.max_word_repetition = max_word_repetition
        self.min_stopword_ratio = min_stopword_ratio
        self.max_symbol_ratio = max_symbol_ratio
        
        # 常见停用词
        self.stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can'
        ])
    
    def filter(self, text: str) -> bool:
        """
        检查文本是否应该保留
        
        返回 True 表示保留，False 表示过滤
        """
        if len(text) < self.min_length:
            return False
        
        words = text.split()
        if len(words) < self.min_words:
            return False
        
        # 词重复检测
        word_counts = {}
        for w in words:
            w_lower = w.lower()
            word_counts[w_lower] = word_counts.get(w_lower, 0) + 1
        
        if word_counts:
            max_rep = max(word_counts.values()) / len(words)
            if max_rep > self.max_word_repetition:
                return False
        
        # 停用词比例
        stopword_count = sum(1 for w in words if w.lower() in self.stopwords)
        if stopword_count / len(words) < self.min_stopword_ratio:
            return False
        
        # 符号比例检测（检测代码或乱码）
        symbol_chars = sum(1 for c in text if c in '#<>{}[]|\\')
        if symbol_chars / len(text) > self.max_symbol_ratio:
            return False
        
        return True


class DocumentDeduplicator:
    """文档去重器"""
    
    def __init__(self, method='minhash', threshold=0.8):
        self.method = method
        self.threshold = threshold
        self.seen_hashes = set()
    
    def compute_hash(self, text: str) -> str:
        """计算文本哈希"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """检查是否重复"""
        doc_hash = self.compute_hash(text)
        
        if doc_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(doc_hash)
        return False


class DataCleaningPipeline:
    """
    完整的数据清洗流水线
    
    流程:
    1. 读取原始数据
    2. 文本清洗
    3. 质量过滤
    4. 去重
    5. 保存
    """
    
    def __init__(self, 
                 num_workers: int = 8,
                 clean_config: dict = None,
                 filter_config: dict = None):
        
        self.num_workers = num_workers
        self.cleaner = TextCleaner()
        self.filter = QualityFilter(**(filter_config or {}))
        self.deduplicator = DocumentDeduplicator()
    
    def process_document(self, doc: Dict) -> Optional[Dict]:
        """处理单个文档"""
        text = doc.get('text', '')
        doc_id = doc.get('id', hashlib.md5(text.encode()).hexdigest())
        
        # 1. 清洗
        text = self.cleaner.clean(text)
        
        # 2. 质量过滤
        if not self.filter.filter(text):
            return None
        
        # 3. 去重
        if self.deduplicator.is_duplicate(text, doc_id):
            return None
        
        return {'id': doc_id, 'text': text}
    
    def process_file(self, input_path: str, output_path: str):
        """处理单个文件"""
        kept = 0
        filtered = 0
        duplicates = 0
        
        with open(output_path, 'w', encoding='utf-8') as out_file:
            with open(input_path, 'r', encoding='utf-8') as in_file:
                for line in tqdm(in_file, desc="Processing"):
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        filtered += 1
                        continue
                    
                    result = self.process_document(doc)
                    
                    if result is None:
                        # 判断是过滤还是去重
                        text = doc.get('text', '')
                        if not self.filter.filter(text):
                            filtered += 1
                        else:
                            duplicates += 1
                    else:
                        out_file.write(json.dumps(result) + '\n')
                        kept += 1
        
        print(f"\n处理完成:")
        print(f"  保留：{kept}")
        print(f"  过滤：{filtered}")
        print(f"  去重：{duplicates}")
        
        return {'kept': kept, 'filtered': filtered, 'duplicates': duplicates}
    
    def process_directory(self, input_dir: str, output_dir: str):
        """处理目录下的所有文件"""
        import os
        import glob
        
        files = glob.glob(os.path.join(input_dir, '*.jsonl'))
        
        for file_path in files:
            output_path = os.path.join(
                output_dir, 
                os.path.basename(file_path).replace('.jsonl', '_cleaned.jsonl')
            )
            self.process_file(file_path, output_path)


# 使用示例
if __name__ == "__main__":
    pipeline = DataCleaningPipeline(
        num_workers=8,
        filter_config={
            'min_length': 200,
            'min_words': 50,
        }
    )
    
    stats = pipeline.process_file('raw_data.jsonl', 'cleaned_data.jsonl')
```

---



## 3.4.2 数据去重技术

### MinHash LSH 去重

```python
# examples/3-4-data/deduplication.py
"""
文档级去重：MinHash + LSH
"""

from datasketch import MinHash, MinHashLSH
from typing import List, Tuple


class MinHashDeduplicator:
    """
    MinHash 去重实现
    
    原理:
    1. 将文档转换为 MinHash 签名
    2. 使用 LSH 快速查找候选重复
    3. 验证 Jaccard 相似度
    """
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        """
        参数:
            num_perm: MinHash 排列数（越大越精确）
            threshold: 相似度阈值
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.doc_ids = []
    
    def _create_minhash(self, text: str) -> MinHash:
        """创建 MinHash"""
        m = MinHash(num_perm=self.num_perm)
        
        # 使用字符 n-gram
        n = 5
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n].encode('utf-8')
            m.update(ngram)
        
        return m
    
    def find_duplicates(self, documents: List[Tuple[str, str]]) -> List[int]:
        """
        查找重复文档
        
        参数:
            documents: [(doc_id, text), ...]
        
        返回:
            重复文档索引列表
        """
        duplicates = []
        
        for idx, (doc_id, text) in enumerate(documents):
            minhash = self._create_minhash(text)
            
            # 查询 LSH 找到候选
            candidates = self.lsh.query(minhash)
            
            if candidates:
                duplicates.append(idx)
            else:
                # 插入 LSH
                self.lsh.insert(doc_id, minhash)
        
        return duplicates


def compute_jaccard_similarity(s1: str, s2: str, n: int = 5) -> float:
    """
    计算 Jaccard 相似度
    
    使用字符 n-gram
    """
    def get_ngrams(s, n):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    
    set1 = get_ngrams(s1, n)
    set2 = get_ngrams(s2, n)
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union


# 使用示例
if __name__ == "__main__":
    docs = [
        ("doc1", "这是一段测试文本用于去重演示"),
        ("doc2", "这是另一段完全不同的内容"),
        ("doc3", "这是一段测试文本用于去重演示"),  # 与 doc1 重复
        ("doc4", "这是新的内容"),
    ]
    
    deduplicator = MinHashDeduplicator(threshold=0.8)
    duplicates = deduplicator.find_duplicates(docs)
    
    print(f"重复文档索引：{duplicates}")
```

---



## 3.4.3 数据质量评估

```python
# examples/3-4-data/quality_assessment.py
"""
数据质量评估指标
"""

import numpy as np
from collections import Counter
import re


class QualityMetrics:
    """文本质量评估"""
    
    def __init__(self):
        self.common_words = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
        ])
    
    def compute_all_metrics(self, text: str) -> dict:
        """计算所有质量指标"""
        return {
            'length': self.length_score(text),
            'word_diversity': self.word_diversity(text),
            'stopword_ratio': self.stopword_ratio(text),
            'symbol_ratio': self.symbol_ratio(text),
            'repetition_score': self.repetition_score(text),
            'perplexity_estimate': self.perplexity_estimate(text),
        }
    
    def length_score(self, text: str) -> float:
        """长度评分"""
        length = len(text.split())
        return min(1.0, length / 500)  # 500 词以上满分
    
    def word_diversity(self, text: str) -> float:
        """词多样性 (TTR)"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def stopword_ratio(self, text: str) -> float:
        """停用词比例"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        stopword_count = sum(1 for w in words if w in self.common_words)
        ratio = stopword_count / len(words)
        
        # 理想比例 0.2-0.5
        if 0.2 <= ratio <= 0.5:
            return 1.0
        elif ratio < 0.2:
            return ratio / 0.2
        else:
            return max(0, 1 - (ratio - 0.5) / 0.5)
    
    def symbol_ratio(self, text: str) -> float:
        """特殊符号比例"""
        special_chars = sum(1 for c in text if c in '#<>{}[]|\\$%^&*')
        ratio = special_chars / len(text) if text else 0
        
        # 符号比例越低越好
        return max(0, 1 - ratio * 10)
    
    def repetition_score(self, text: str) -> float:
        """重复检测"""
        words = text.lower().split()
        if not words:
            return 1.0
        
        word_counts = Counter(words)
        max_count = max(word_counts.values())
        
        # 最大重复不超过 10%
        return max(0, 1 - (max_count / len(words) - 0.1) * 2)
    
    def perplexity_estimate(self, text: str) -> float:
        """
        简化困惑度估计
        基于词频的近似
        """
        words = text.lower().split()
        
        # 常见词<unk>少 → 质量好
        known_ratio = sum(1 for w in words if w in self.common_words) / len(words)
        
        return known_ratio  # 越高越好


def score_dataset(file_path: str, output_path: str):
    """评估整个数据集"""
    metrics = QualityMetrics()
    
    scores = []
    
    with open(file_path, 'r') as f, open(output_path, 'w') as out:
        for line in f:
            doc = json.loads(line)
            text = doc.get('text', '')
            
            score_dict = metrics.compute_all_metrics(text)
            
            # 综合评分
            total_score = np.mean(list(score_dict.values()))
            score_dict['total'] = total_score
            
            scores.append(total_score)
            
            # 保存带评分的数据
            doc['quality_scores'] = score_dict
            out.write(json.dumps(doc) + '\n')
    
    print(f"平均质量评分：{np.mean(scores):.3f}")
    print(f"评分标准差：{np.std(scores):.3f}")
    
    return scores
```

---



## 3.4.4 数据混合策略

```python
# examples/3-4-data/data_mixing.py
"""
训练数据混合策略
"""

from typing import Dict, List
import numpy as np


class DataMixer:
    """
    多源数据混合器
    
    支持:
    - 固定比例混合
    - 动态调整混合
    - 课程学习混合
    """
    
    def __init__(self, data_sources: Dict[str, str], 
                 proportions: Dict[str, float]):
        """
        参数:
            data_sources: {name: filepath}
            proportions: {name: target_ratio}
        """
        self.sources = data_sources
        self.proportions = proportions
        self.data_buffers = {name: [] for name in data_sources}
        self.exhausted = {name: False for name in data_sources}
    
    def sample(self, total_tokens: int) -> str:
        """按混合比例采样"""
        # 确定从哪个源采样
        choices = [name for name, p in self.proportions.items() 
                   if not self.exhausted.get(name, False)]
        
        if not choices:
            return None
        
        probs = [self.proportions[name] for name in choices]
        probs = np.array(probs) / sum(probs)
        
        chosen = np.random.choice(choices, p=probs)
        
        # 从选中的源读取数据
        # 实现省略...
        
        return chosen


# 推荐的数据混合比例
RECOMMENDED_MIX = {
    # 通用预训练
    "pretrain_general": {
        "web_crawl": 0.60,      # CommonCrawl 等
        "books": 0.15,          # 书籍
        "wikipedia": 0.10,      # 百科
        "code": 0.10,           # GitHub 代码
        "news": 0.05,           # 新闻文章
    },
    
    # 代码模型
    "pretrain_code": {
        "code": 0.70,
        "technical_docs": 0.15,
        "stackoverflow": 0.10,
        "general": 0.05,
    },
    
    # 对话模型
    "pretrain_dialogue": {
        "dialogue": 0.40,
        "social_media": 0.20,
        "books": 0.15,
        "wikipedia": 0.10,
        "web": 0.15,
    },
}
```

---



## 练习题

1. **质量阈值选择**：如何确定适合的质量过滤阈值？

2. **去重影响**：去重对模型训练有什么正面/负面影响？

3. **混合策略**：多语言模型应该如何混合各语言数据？

---

[← 上一节：3.3 推理优化](3-3-inference.md) | [下一节：3.5 MLOps →](3-5-mlops.md)
