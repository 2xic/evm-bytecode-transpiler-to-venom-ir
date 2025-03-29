"""
We need a small symbolic EVM to be able to handle the lookups
"""

from copy import deepcopy
from typing import List

"""
TODO: This should replace the existing traces type
"""


class ExecutionTrace:
	def __init__(self, blocks=[]):
		self.blocks: List[int] = blocks

	def __str__(self):
		return f"{self.blocks}"

	def __repr__(self):
		return self.__str__()


# For each
class ProgramTrace:
	def __init__(self):
		self.execution = ExecutionTrace()
		self.traces: List[ExecutionTrace] = []

	def block_traces(self, block_id) -> List[ExecutionTrace]:
		traces: List[ExecutionTrace] = []
		for i in self.traces:
			if block_id in i.blocks:
				traces.append(i)
		return traces

	def fork(self):
		# Each time there is a conditional block we need to fork the trace.
		self.traces.append(self.execution)
		self.execution = ExecutionTrace(deepcopy(self.execution.blocks))
		return self.execution

	def create(self):
		return self.execution


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
	def __init__(self, id, opcode, inputs, pc, block):
		super().__init__(id, pc)
		self.opcode = opcode
		self.inputs = inputs
		self.pc = pc
		self.block = block

	def constant_fold(self):
		return self

	def __str__(self):
		name = self.__class__.__name__
		return f"{name}({self.opcode}, {self.inputs})\tpc: {self.pc}"

	def __repr__(self):
		return self.__str__()


class SymbolicPcOpcode(SymbolicOpcode):
	def __init__(self, id, opcode, inputs, pc, block):
		super().__init__(id, opcode, inputs, pc, block)

	def constant_fold(self):
		return ConstantValue(-1, self.pc, None)


class SymbolicAndOpcode(SymbolicOpcode):
	def __init__(self, id, opcode, inputs, pc, block):
		super().__init__(id, opcode, inputs, pc, block)

	def constant_fold(self):
		[a, b] = self.inputs
		return ConstantValue(
			-1, a.constant_fold().value & b.constant_fold().value, None
		)


class ConstantValue(SymbolicValue):
	def __init__(self, id, value, block):
		super().__init__(id, block)
		self.value = value
		self.block = block

	def constant_fold(self):
		return self


class EVM:
	def __init__(self, pc):
		self.stack = []
		self.pc = pc
		self.trace_id = 0

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
