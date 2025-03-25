"""
Implements a version of algorithm 2 from https://c9x.me/compile/bib/braun13cc.pdf

How is this going to work when we run things symbolically?
1, We know the location of the variables.
2. 


"""
from ordered_set import OrderedSet


class PhiFunction:
    def __init__(self, block):
        self.operands = []

class VariableStore:
    def __init__(self):
        self.store = {}
        self.sealed_block = OrderedSet()
        self.incomplete_phis = {}

    def write(self, variable, block, value):
        pass

    def read(self, variable, block):
        if variable in self.store[block]:
            return self.store[block][variable]
        return self.read_recursive(variable, block)
    
    def read_recursive(self, variable, block):
        if block in self.sealed_block:
            value = PhiFunction(block)
            self.incomplete_phis[block][variable] = value
        elif len(block.parents) == 1:
            value = self.read(variable, block)
        else:
            value = PhiFunction(block)
            self.write(variable, block, value)
            value = self.add_phi_operands(variable, value)
        self.write(variable, block, value)
        return value
    
    def add_phi_operands(self, phi):
        for block in block.incomming:
            phi.op
