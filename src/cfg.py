from test_utils.compiler import SolcCompiler
from blocks import get_calling_blocks
import graphviz
from opcodes import get_opcodes_from_bytes, PushOpcode

code = """
contract Hello {
	function test() public returns (uint256) {
		return bagel();
	}

	function bagel() public returns (uint256) {
		return 1;
	}
}
"""
output = SolcCompiler().compile(code)
cfg = get_calling_blocks(get_opcodes_from_bytes(output))
dot = graphviz.Digraph(comment='cfg', format='png')

for cfg_block in cfg.blocks:
	block = []
	for opcode in cfg_block.opcodes:
		block.append(f"{hex(opcode.pc)}: {opcode} \\l")
	dot.node(hex(cfg_block.start_offset), "".join(block), shape="box")
	for edge in cfg_block.outgoing:
		dot.edge(hex(cfg_block.start_offset), hex(edge))

dot.render("test", cleanup=True)
