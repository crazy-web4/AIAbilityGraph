# 6.1 Python 编程进阶

> 掌握 Python 高级编程技巧，提升算法开发效率和代码质量。

## 学习目标

- [ ] 掌握 Python 性能优化技巧
- [ ] 理解异步编程模型
- [ ] 学会使用类型提示和泛型
- [ ] 掌握元编程技巧

---

## 6.1.1 性能优化

### 1. 使用高效数据结构

```python
# examples/6-1-python/performance_tips.py
"""
Python 性能优化技巧
"""

from collections import deque, defaultdict, Counter
from typing import List, Set, Dict
import timeit
import sys

# ========== 1. 选择合适的数据结构 ==========

# ❌ 低效：列表查找 O(n)
def find_item_linear(items: List, target):
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1

# ✅ 高效：集合查找 O(1)
def find_item_set(items: Set, target):
    return target in items

# ========== 2. 使用 deque 进行队列操作 ==========

# ❌ 低效：列表 pop(0) 是 O(n)
def process_queue_list(items):
    queue = list(items)
    while queue:
        item = queue.pop(0)  # O(n)
        process(item)

# ✅ 高效：deque popleft() 是 O(1)
def process_queue_deque(items):
    queue = deque(items)
    while queue:
        item = queue.popleft()  # O(1)
        process(item)

# ========== 3. 使用 Counter 统计 ==========

# ❌ 低效：手动计数
def count manually(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts

# ✅ 高效：Counter
def count_with_counter(items):
    return Counter(items)

# ========== 4. 使用生成器节省内存 ==========

# ❌ 内存占用大：一次性加载所有数据
def sum_all_file_lines(filepath):
    with open(filepath) as f:
        lines = f.readlines()  # 全部加载到内存
        return sum(len(line) for line in lines)

# ✅ 内存高效：生成器逐行读取
def sum_all_lines_generator(filepath):
    with open(filepath) as f:
        return sum(len(line) for line in f)  # 逐行迭代

# ========== 5. 使用局部变量 ==========

def slow_function():
    # 全局/属性查找较慢
    result = []
    for i in range(1000):
        result.append(str(i))
    return ''.join(result)

def fast_function():
    # 局部变量更快
    append = result.append  # 缓存方法引用
    result = []
    for i in range(1000):
        append(str(i))
    return ''.join(result)

# ========== 6. 字符串拼接 ==========

# ❌ 低效：字符串不可变，每次创建新对象
def concat_slow(words):
    result = ""
    for word in words:
        result += word
    return result

# ✅ 高效：join 只创建一次
def concat_fast(words):
    return "".join(words)


# ========== 性能对比测试 ==========

if __name__ == "__main__":
    # 列表 vs 集合查找
    items_list = list(range(10000))
    items_set = set(items_list)
    
    print("查找性能对比 (1000 次查找):")
    print(f"  列表：{timeit.timeit(lambda: 9999 in items_list, number=1000):.4f}s")
    print(f"  集合：{timeit.timeit(lambda: 9999 in items_set, number=1000):.4f}s")
    
    # 字符串拼接
    words = ["word"] * 1000
    print("\n字符串拼接对比:")
    print(f"  += 方式：{timeit.timeit(lambda: concat_slow(words), number=100):.4f}s")
    print(f"  join 方式：{timeit.timeit(lambda: concat_fast(words), number=1000):.4f}s")
```

---

## 6.1.2 异步编程

```python
# examples/6-1-python/async_programming.py
"""
异步编程最佳实践
"""

import asyncio
import aiohttp
from typing import List, AsyncIterator
from concurrent.futures import ThreadPoolExecutor

# ========== 基础异步函数 ==========

async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    """异步获取 URL 内容"""
    async with session.get(url) as response:
        return await response.text()

async def fetch_all_urls(urls: List[str]) -> List[str]:
    """并发获取多个 URL"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# ========== 异步生成器 ==========

async def async_range(start: int, end: int, delay: float = 0) -> AsyncIterator[int]:
    """异步生成器"""
    for i in range(start, end):
        if delay:
            await asyncio.sleep(delay)
        yield i

async def consume_async_iterator():
    """消费异步生成器"""
    async for num in async_range(1, 6, delay=0.1):
        print(f"收到：{num}")

# ========== 生产者 - 消费者模式 ==========

async def producer(queue: asyncio.Queue, n_items: int):
    """生产者"""
    for i in range(n_items):
        await queue.put(f"item-{i}")
        print(f"生产：item-{i}")
    await queue.put(None)  # 结束标记

async def consumer(queue: asyncio.Queue):
    """消费者"""
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"消费：{item}")
        await asyncio.sleep(0.1)  # 模拟处理
        queue.task_done()

async def run_producer_consumer():
    """运行生产者 - 消费者"""
    queue = asyncio.Queue(maxsize=5)
    
    await asyncio.gather(
        producer(queue, 20),
        consumer(queue),
        consumer(queue)  # 多个消费者
    )

# ========== CPU 密集型任务的异步处理 ==========

def cpu_intensive_task(n: int) -> int:
    """CPU 密集型计算"""
    return sum(i * i for i in range(n))

async def async_cpu_task(n: int) -> int:
    """在后台线程池运行 CPU 密集型任务"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, cpu_intensive_task, n)

# ========== 使用示例 ==========

async def main():
    # 并发获取 URL
    urls = [
        "https://api.github.com",
        "https://api.example.com",
        "https://httpbin.org/get"
    ]
    results = await fetch_all_urls(urls)
    
    # 生产者 - 消费者
    await run_producer_consumer()
    
    # CPU 密集型任务
    result = await async_cpu_task(10_000_000)
    print(f"计算结果：{result}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6.1.3 类型提示进阶

```python
# examples/6-1-python/type_hints.py
"""
Python 类型提示进阶用法
"""

from typing import (
    TypeVar, Generic, Protocol, Callable,
    TypedDict, Literal, overload, Union,
    Optional, Type, NewType
)
from dataclasses import dataclass

# ========== TypeVar 和泛型 ==========

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class Stack(Generic[T]):
    """泛型栈"""
    
    def __init__(self) -> None:
        self.items: List[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)
    
    def pop(self) -> T:
        return self.items.pop()
    
    def is_empty(self) -> bool:
        return len(self.items) == 0

# ========== Protocol (结构子类型) ==========

class Drawable(Protocol):
    """可绘制协议"""
    
    def draw(self) -> None: ...
    def get_size(self) -> tuple: ...

def render_all(items: List[Drawable]) -> None:
    """渲染所有可绘制对象"""
    for item in items:
        item.draw()

# ========== TypedDict ==========

class UserEvent(TypedDict):
    """用户事件类型"""
    user_id: int
    action: str
    timestamp: str
    metadata: Optional[dict]

def process_event(event: UserEvent) -> None:
    print(f"用户 {event['user_id']} 执行了 {event['action']}")

# ========== Literal 类型 ==========

from typing import Literal

def set_log_level(level: Literal['DEBUG', 'INFO', 'WARN', 'ERROR']) -> None:
    """设置日志级别"""
    print(f"日志级别设置为：{level}")

# 类型检查器会拒绝以下调用:
# set_log_level('INVALID')  # Error!

# ========== overload 装饰器 ==========

@overload
def process(data: str) -> str: ...

@overload
def process(data: int) -> int: ...

def process(data: Union[str, int]) -> Union[str, int]:
    """重载函数"""
    if isinstance(data, str):
        return data.upper()
    return data * 2

# ========== NewType 创建类型别名 ==========

UserId = NewType('UserId', int)
ProductId = NewType('ProductId', str)

def get_user(user_id: UserId) -> Optional[dict]:
    """根据用户 ID 获取用户"""
    print(f"查询用户：{user_id}")
    return {"id": user_id, "name": "Test"}

# ========== Callable 类型 ==========

Callback = Callable[[str, int], bool]

def register_callback(
    name: str, 
    callback: Callable[[str, int], bool]
) -> None:
    """注册回调函数"""
    print(f"注册回调：{name}")

# ========== 类型守卫 ==========

from typing import TypeGuard

def is_string_list(items: list) -> TypeGuard[list[str]]:
    """类型守卫：检查是否为字符串列表"""
    return all(isinstance(item, str) for item in items)

def process_strings(items: list) -> None:
    if is_string_list(items):
        # 类型检查器现在知道 items 是 list[str]
        result = "".join(items)
        print(result)

# ========== 数据类类型注解 ==========

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    device: str = "cuda"
    
    def validate(self) -> bool:
        if self.learning_rate <= 0:
            raise ValueError("学习率必须大于 0")
        return True


# ========== 使用示例 ==========

def demo_type_hints():
    # 泛型栈
    int_stack = Stack[int]()
    int_stack.push(1)
    int_stack.push(2)
    print(f"栈顶元素：{int_stack.pop()}")
    
    # Protocol
    class Circle:
        def draw(self): print("绘制圆形")
        def get_size(self): return (10, 10)
    
    render_all([Circle()])
    
    # TypedDict
    event: UserEvent = {
        "user_id": 123,
        "action": "login",
        "timestamp": "2024-01-01T00:00:00Z",
        "metadata": None
    }
    process_event(event)
    
    # 类型守卫
    items = ["hello", "world"]
    process_strings(items)


if __name__ == "__main__":
    demo_type_hints()
```

---

## 6.1.4 元编程技巧

```python
# examples/6-1-python/metaprogramming.py
"""
元编程：代码生成代码
"""

from functools import wraps
from typing import Any, Callable
import time

# ========== 装饰器 ==========

def timer(func: Callable) -> Callable:
    """计时装饰器"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} 耗时：{elapsed:.4f}s")
        return result
    
    return wrapper

def retry(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # 指数退避
            raise last_exception
        
        return wrapper
    return decorator

# ========== 类装饰器 ==========

def singleton(cls):
    """单例装饰器"""
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class DatabaseConnection:
    """单例数据库连接"""
    def __init__(self):
        print("初始化数据库连接")

# ========== 元类 ==========

class SingletonMeta(type):
    """单例元类"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Config(metaclass=SingletonMeta):
    """单例配置类"""
    def __init__(self):
        self.settings = {}

# ========== 动态创建类 ==========

def create_model_class(
    name: str,
    fields: dict,
    methods: dict = None
) -> type:
    """
    动态创建模型类
    
    参数:
        name: 类名
        fields: 字段定义
        methods: 方法定义
    """
    def __init__(self, **kwargs):
        for field, default in fields.items():
            setattr(self, field, kwargs.get(field, default))
    
    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{name}({attrs})"
    
    # 创建类
    class_dict = {
        '__init__': __init__,
        '__repr__': __repr__,
        **(methods or {})
    }
    
    return type(name, (object,), class_dict)

# ========== 描述符 ==========

class ValidatedField:
    """验证描述符"""
    
    def __init__(self, validator: Callable[[Any], bool]):
        self.validator = validator
        self.data = {}
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.data.get(id(obj))
    
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(f"验证失败：{value}")
        self.data[id(obj)] = value

class Person:
    """使用描述符验证的类"""
    age = ValidatedField(lambda x: x >= 0)
    email = ValidatedField(lambda x: '@' in x)
    
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

# ========== 使用示例 ==========

@timer
def slow_function():
    time.sleep(0.1)

@retry(max_retries=3, delay=0.1)
def unreliable_function():
    import random
    if random.random() < 0.5:
        raise Exception("随机失败")
    return "成功"

def main():
    # 装饰器
    slow_function()
    
    # 单例
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    print(f"db1 is db2: {db1 is db2}")  # True
    
    # 动态创建类
    User = create_model_class('User', {
        'id': 0,
        'name': '',
        'email': ''
    })
    
    user = User(id=1, name="Jack", email="jack@example.com")
    print(user)
    
    # 描述符验证
    try:
        person = Person("Jack", 25, "jack@example.com")
        print(f"创建成功：{person}")
        
        # 这会失败
        person.age = -1
    except ValueError as e:
        print(f"验证错误：{e}")


if __name__ == "__main__":
    main()
```

---

## 练习题

1. **性能优化**：找出以下代码的性能瓶颈并优化:
```python
def process_data(data):
    result = []
    for item in data:
        if item not in result:
            result.append(item)
    return result
```

2. **异步编程**：实现一个异步批处理函数，限制并发数为 N。

3. **类型提示**：为一个 LRU 缓存类添加完整的类型注解。

---

[← 章前导引](README.md) | [下一节：6.2 算法工程化 →](6-2-engineering.md)
