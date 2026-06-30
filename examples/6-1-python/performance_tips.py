#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 性能优化技巧

涵盖:
- 高效数据结构
- 字符串处理优化
- 生成器与迭代器
- 局部变量优化
"""

from collections import deque, defaultdict, Counter
from typing import List, Set
import timeit
import sys


# ========== 1. 选择合适的数据结构 ==========

def find_item_linear(items: List, target):
    """❌ 低效：列表查找 O(n)"""
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1


def find_item_set(items: Set, target):
    """✅ 高效：集合查找 O(1)"""
    return target in items


# ========== 2. 使用 deque 进行队列操作 ==========

def process_queue_list(items):
    """❌ 低效：列表 pop(0) 是 O(n)"""
    queue = list(items)
    results = []
    while queue:
        item = queue.pop(0)  # O(n)
        results.append(item * 2)
    return results


def process_queue_deque(items):
    """✅ 高效：deque popleft() 是 O(1)"""
    queue = deque(items)
    results = []
    while queue:
        item = queue.popleft()  # O(1)
        results.append(item * 2)
    return results


# ========== 3. 使用 Counter 统计 ==========

def count_manually(items):
    """❌ 低效：手动计数"""
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def count_with_counter(items):
    """✅ 高效：Counter"""
    return Counter(items)


# ========== 4. 使用生成器节省内存 ==========

def sum_all_file_lines_readall(filepath):
    """❌ 内存占用大：一次性加载所有数据"""
    with open(filepath) as f:
        lines = f.readlines()  # 全部加载到内存
        return sum(len(line) for line in lines)


def sum_all_lines_generator(filepath):
    """✅ 内存高效：生成器逐行读取"""
    with open(filepath) as f:
        return sum(len(line) for line in f)  # 逐行迭代


# ========== 5. 使用局部变量 ==========

def slow_function():
    """❌ 全局/属性查找较慢"""
    result = []
    for i in range(1000):
        result.append(str(i))
    return ''.join(result)


def fast_function():
    """✅ 局部变量更快"""
    result = []
    append = result.append  # 缓存方法引用
    for i in range(1000):
        append(str(i))
    return ''.join(result)


# ========== 6. 字符串拼接 ==========

def concat_slow(words):
    """❌ 低效：字符串不可变，每次创建新对象"""
    result = ""
    for word in words:
        result += word
    return result


def concat_fast(words):
    """✅ 高效：join 只创建一次"""
    return "".join(words)


# ========== 7. 列表推导式 vs 循环 ==========

def squares_loop(n):
    """❌ 传统循环"""
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(i * i)
    return result


def squares_comprehension(n):
    """✅ 列表推导式"""
    return [i * i for i in range(n) if i % 2 == 0]


# ========== 8. 字符串连接 ==========

def join_strings_slow(strings):
    """❌ 低效：+= 连接"""
    result = ""
    for s in strings:
        result += s
    return result


def join_strings_fast(strings):
    """✅ 高效：join"""
    return "".join(strings)


# ========== 性能对比测试 ==========

def run_benchmarks():
    """运行性能对比测试"""
    print("=" * 60)
    print("Python 性能优化对比测试")
    print("=" * 60)
    print()

    # 1. 列表 vs 集合查找
    print("【1. 查找性能对比】")
    items_list = list(range(10000))
    items_set = set(items_list)

    time_list = timeit.timeit(
        lambda: 9999 in items_list,
        number=1000
    )
    time_set = timeit.timeit(
        lambda: 9999 in items_set,
        number=1000
    )

    print(f"  列表查找 (1000 次): {time_list:.4f}s")
    print(f"  集合查找 (1000 次): {time_set:.4f}s")
    print(f"  加速比：{time_list/time_set:.1f}x")
    print()

    # 2. 队列操作
    print("【2. 队列操作对比】")
    test_data = list(range(1000))

    time_list_queue = timeit.timeit(
        lambda: process_queue_list(test_data),
        number=100
    )
    time_deque_queue = timeit.timeit(
        lambda: process_queue_deque(test_data),
        number=100
    )

    print(f"  list.pop(0): {time_list_queue:.4f}s")
    print(f"  deque.popleft(): {time_deque_queue:.4f}s")
    print(f"  加速比：{time_list_queue/time_deque_queue:.1f}x")
    print()

    # 3. 计数
    print("【3. 计数操作对比】")
    test_items = ['a', 'b', 'c', 'a', 'b', 'a'] * 1000

    time_manual = timeit.timeit(
        lambda: count_manually(test_items),
        number=100
    )
    time_counter = timeit.timeit(
        lambda: count_with_counter(test_items),
        number=100
    )

    print(f"  手动计数：{time_manual:.4f}s")
    print(f"  Counter: {time_counter:.4f}s")
    print()

    # 4. 字符串拼接
    print("【4. 字符串拼接对比】")
    words = ["word"] * 1000

    time_slow = timeit.timeit(
        lambda: concat_slow(words),
        number=100
    )
    time_fast = timeit.timeit(
        lambda: concat_fast(words),
        number=1000
    )

    print(f"  += 方式 (100 次): {time_slow:.4f}s")
    print(f"  join 方式 (1000 次): {time_fast:.4f}s")
    print()

    # 5. 局部变量优化
    print("【5. 局部变量优化】")
    time_slow = timeit.timeit(slow_function, number=100)
    time_fast = timeit.timeit(fast_function, number=100)

    print(f"  原始方式：{time_slow:.4f}s")
    print(f"  优化方式：{time_fast:.4f}s")
    print(f"  提升：{(time_slow-time_fast)/time_slow*100:.1f}%")
    print()

    # 6. 列表推导式
    print("【6. 列表推导式对比】")
    time_loop = timeit.timeit(lambda: squares_loop(1000), number=1000)
    time_comp = timeit.timeit(lambda: squares_comprehension(1000), number=1000)

    print(f"  传统循环：{time_loop:.4f}s")
    print(f"  推导式：{time_comp:.4f}s")
    print(f"  提升：{(time_loop-time_comp)/time_loop*100:.1f}%")
    print()

    print("=" * 60)


# ========== 内存使用对比 ==========

def compare_memory_usage():
    """比较不同数据结构的内存使用"""
    print("\n【内存使用对比】")

    # 列表 vs 生成器
    list_comp = [i * i for i in range(1000000)]
    gen_exp = (i * i for i in range(1000000))

    list_size = sys.getsizeof(list_comp)
    gen_size = sys.getsizeof(gen_exp)

    print(f"  列表 [i² for i in 1..1000000]: {list_size / 1024 / 1024:.2f} MB")
    print(f"  生成器 (i² for i in 1..1000000): {gen_size / 1024 / 1024:.4f} MB")
    print(f"  内存节省：{(1 - gen_size/list_size) * 100:.2f}%")


if __name__ == "__main__":
    run_benchmarks()
    compare_memory_usage()

    print("\n【优化建议总结】")
    print("  1. 查找操作多用 set 和 dict，少用 list")
    print("  2. 队列操作使用 deque 而非 list")
    print("  3. 计数用 Counter，排序用 sorted()")
    print("  4. 大数据用生成器，避免一次性加载")
    print("  5. 循环内缓存方法引用")
    print("  6. 字符串拼接用 join 而非 +=")
    print("  7. 优先使用列表推导式")
