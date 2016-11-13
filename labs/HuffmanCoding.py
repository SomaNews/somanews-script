import heapq
from collections import defaultdict
import pandas as pd

# https://gist.github.com/nboubakr/0eec4ea650eeb6dc21f9

class HuffmanCoding():
    def __init__(self, text):
        self.text_ = text
        self.frequency_ = defaultdict(int)
        for symbol in self.text_:
            self.frequency_[symbol] += 1
    
    def encode(self):
        heap = [[weight, [symbol, '']] for symbol, weight in self.frequency_.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        self.huff_ = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
        self.huff_dict_ = {p[0]: {"code":p[1],"weight":self.frequency_[p[0]]} for p in self.huff_}
        return self.huff_dict_
    
    def print_huff(self):
        print "Symbol".ljust(10) + "Weight".ljust(10) + "Huffman Code"
        for p in self.huff_:
            print p[0].ljust(10) + str(self.frequency_[p[0]]).ljust(10) + p[1]
            
    def decode(self, targets):
        results = []
        for target in targets:
            code = ''
            weight = 0
            for s in target:
                weight = weight + self.huff_dict_[s]['weight']
                code = code + self.huff_dict_[s]['code']
            results.append((target, weight, len(code), code))
        return pd.DataFrame(results, columns=['word', 'weight', 'len', 'code'])