from blocks import get_calling_blocks
import graphviz
from opcodes import get_opcodes_from_bytes, PushOpcode

cfg = get_calling_blocks(
	get_opcodes_from_bytes(bytes.fromhex("6080604052348015600e575f80fd5b50600436106026575f3560e01c8063f8a8fd6d14602a575b5f80fd5b60306044565b604051603b91906062565b60405180910390f35b5f6001905090565b5f819050919050565b605c81604c565b82525050565b5f60208201905060735f8301846055565b9291505056"))
)
dot = graphviz.Digraph(comment='cfg', format='png')

for cfg_block in cfg.blocks:
	block = []
	for opcode in cfg_block.opcodes:
		block.append(f"{hex(opcode.pc)}: {opcode} \\l")
	dot.node(hex(cfg_block.start_offset), "".join(block), shape="box")
	for edge in cfg_block.outgoing:
		dot.edge(hex(cfg_block.start_offset), hex(edge))

dot.render("test", cleanup=True)
