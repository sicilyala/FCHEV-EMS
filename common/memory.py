import random
from collections import deque  # 两端都可以操作的序列
import numpy as np

class MemoryBuffer:
    def __init__(self, args):
        self.maxSize = int(args.buffer_size)
        self.batchSize = int(args.batch_size)
        self.buffer = deque(maxlen=self.maxSize)    # 限制长度，左侧数据自动被删除
        self.currentSize = 0
        self.counter = 0

    # def store(self, s, a, r, s_):
        # transition = (s, a, r, s_)
    def store(self, s, a, r, s_, mask):
        transition = (s, a, r, s_, mask)
        self.buffer.append(transition)
        self.counter += 1
        self.currentSize = min(self.counter, self.maxSize)
        
    def random_sample(self):
        batch = random.sample(self.buffer, self.batchSize)
        
        s = np.float32([arr[0] for arr in batch])
        a = np.float32([arr[1] for arr in batch])
        r = np.float32([arr[2] for arr in batch])
        s_ = np.float32([arr[3] for arr in batch])
        mask = np.float32([arr[4] for arr in batch])
        
        transition = (s, a, r, s_, mask)
        return transition
    
    # priority experience replay technical
    def priority_sample(self):
        
        pass
