"""
We need a small symbolic EVM to be able to handle the lookups 
"""
from opcodes import PushOpcode, DupOpcode
from copy import deepcopy
import random

class SymbolicValue:
	def __init__(self, id, pc):
		self.id = id
		self.pc = pc
		# TODO: this shouldn't really be used
		self.value = 1337

	def __str__(self):
		return f"SymbolicValue({self.id})"

	def __repr__(self):
		return self.__str__()


class SymbolicOpcode(SymbolicValue):
	def __init__(self, opcode, inputs, pc):
		super().__init__(random.randint(0, 256), pc)
		#self.id = id
		self.opcode = opcode
		self.inputs = inputs
		self.pc = pc

	def __str__(self):
		return f"SymbolicOpcode({self.opcode}, {self.inputs})"

	def __repr__(self):
		return self.__str__()

class ConstantValue(SymbolicValue):
	def __init__(self, id, value, pc):
		super().__init__(id, pc)
		self.value = value

	def __str__(self):
		return f"ConstantValue({hex(self.value)})"	

class EVM:
	def __init__(self, pc):
		self.stack = []
		self.pc = pc

	def step(self):
		self.pc += 1

	def peek(self):
		return self.stack[-1]

	def get_item(self, n):
		if len(self.stack) < abs(n):
			return ConstantValue(0, 1337, -1)
		else:
			return self.stack[n]
		
	def swap(self, n):
		index_a = len(self.stack) - 1
		index_b = index_a - n
		stack = self.stack
		assert index_a >= 0
		assert index_b >= 0
		stack[index_a], stack[index_b] = stack[index_b], stack[index_a]
		return self
	
	def dup(self, n):
		var_copy = self.get_item(-n)
		self.stack.append(var_copy)
		return self

	def pop_item(self):
		if len(self.stack) > 0:
			return self.stack.pop()
		return ConstantValue(0, 1337, -1)

	def clone(self):
		return deepcopy(self)


