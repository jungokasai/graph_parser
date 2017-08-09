#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division 
from __future__ import print_function
from collections import defaultdict


class Tarjan:
    def __init__(self, prediction, tokens):
        self._edges = defaultdict(set)
        self._vertices = set((0,))
        for dep, head in enumerate(prediction[tokens]):
            self._vertices.add(dep+1)
            self._edges[head].add(dep+1)
        self._indices = {}
        self._lowlinks = {}
        self._onstack = defaultdict(lambda: False)
        self._SCCs = []
        
        index = 0
        stack = []
        for v in self.vertices:
            if v not in self.indices:
                self.strongconnect(v, index, stack)
    
    def strongconnect(self, v, index, stack):
        """"""
        
        self._indices[v] = index
        self._lowlinks[v] = index
        index += 1
        stack.append(v)
        self._onstack[v] = True
        for w in self.edges[v]:
            if w not in self.indices:
                self.strongconnect(w, index, stack)
                self._lowlinks[v] = min(self._lowlinks[v], self._lowlinks[w])
            elif self._onstack[w]:
                self._lowlinks[v] = min(self._lowlinks[v], self._indices[w])
        
        if self._lowlinks[v] == self._indices[v]:
            self._SCCs.append(set())
            while stack[-1] != v:
                w = stack.pop()
                self._onstack[w] = False
                self._SCCs[-1].add(w)
            w = stack.pop()
            self._onstack[w] = False
            self._SCCs[-1].add(w)
        return
    
    @property
    def edges(self):
        return self._edges
    @property
    def vertices(self):
        return self._vertices
    @property
    def indices(self):
        return self._indices
    @property
    def SCCs(self):
        return self._SCCs
if __name__ == '__main__':
    import numpy as np
    tokens = np.arange(1, 6)
    predictions = np.array([0, 2, 0, 5, 3, 4])
    test = Tarjan(predictions, tokens)
    print(test._SCCs)
    print(test._edges)
    print(test._vertices)
