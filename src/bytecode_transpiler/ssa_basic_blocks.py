from dataclasses import dataclass
from bytecode_transpiler.symbolic import ConstantValue
from ordered_set import OrderedSet


@dataclass
class Block:
	id: ConstantValue

	def __str__(self):
		return str(VyperBlock(self.id.value))

	def __hash__(self):
		return self.id.value


@dataclass(frozen=True)
class VyperBlock:
	id: int

	def __str__(self):
		return f"@{self.tag()}"

	def tag(self):
		return f"block_{hex(self.id)}"


@dataclass(frozen=True)
class VyperVarRef:
	ref: str

	def __str__(self):
		return f"%{self.ref}"


@dataclass(frozen=True)
class VyperBlockRef:
	ref: VyperBlock

	def __str__(self):
		return f"%{self.ref.tag()}"


class VyperPhiRef(VyperVarRef):
	def __str__(self):
		return f"%phi{self.ref}"


@dataclass(frozen=True)
class VyperVariable:
	id: VyperVarRef
	value: str

	def __str__(self):
		return f"{self.id} = {self.value}"


@dataclass
class PhiCounter:
	value: int

	def increment(self):
		old = self.value
		self.value += 1
		return old


@dataclass(frozen=True)
class PhiEdge:
	block: str
	value: str

	def __str__(self):
		# TODO: Improve the logic so that this isn't needed.
		if self.block is None:
			return f"?`, {self.value}"
		else:
			assert self.block is not None
			block = f"{VyperBlock(self.block)}" if self.block > 0 else "@global"
			return f"{block}, {self.value}"


@dataclass
class PhiFunction:
	edge: OrderedSet[PhiEdge]

	def add_edge(self, edge):
		self.edge.append(edge)
		return self

	@property
	def can_skip(self):
		values = OrderedSet([])
		for i in self.edge:
			values.add(i.value)
		return len(self.edge) <= 1 or len(values) <= 1

	def __str__(self):
		return ", ".join(list(map(str, self.edge)))
