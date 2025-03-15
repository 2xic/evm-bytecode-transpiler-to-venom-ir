from test_utils.compiler import SolcCompiler
from opcodes import get_opcodes_from_bytes
from blocks import get_calling_blocks
import graphviz

def create_cfg(bytecode):
	cfg = get_calling_blocks(get_opcodes_from_bytes(bytecode))
	return cfg

def save_cfg(bytecode, name="cfg.png", render=False):
	cfg = create_cfg(bytecode)
	dot = graphviz.Digraph(comment='cfg', format='png')

	for cfg_block in cfg.blocks:
		block = []
		for opcode in cfg_block.opcodes:
			block.append(f"{hex(opcode.pc)}: {opcode} \\l")
		if cfg_block.mark:
			dot.node(hex(cfg_block.start_offset), "".join(block), shape="box", color="red")
		else:
			dot.node(hex(cfg_block.start_offset), "".join(block), shape="box")
		for edge in cfg_block.outgoing:
			dot.edge(hex(cfg_block.start_offset), hex(edge))

	dot.render(name.replace(".png",""), cleanup=True)

if __name__ == "__main__":
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
	print(SolcCompiler().compile(code, via_ir=False).hex())
	save_cfg(SolcCompiler().compile(code, via_ir=False))
