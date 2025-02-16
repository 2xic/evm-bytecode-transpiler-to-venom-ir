from test_utils.compiler import SolcCompiler
from opcodes import get_opcodes_from_bytes
from blocks import get_calling_blocks, flatten_blocks
import graphviz

def create_cfg(bytecode, name="cfg.png", flatten=False):
	cfg = get_calling_blocks(get_opcodes_from_bytes(bytecode))
	dot = graphviz.Digraph(comment='cfg', format='png')

	# Flatten the CFG will create duplicates nodes to not have infinitive loops which is not compatible with 
	# SSA
	if flatten:
		cfg = flatten_blocks(cfg)


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
    contract Counter {
        int private count = 0;


        function _getCount() internal view returns (int) {
            return count;
        }

        function getCount() public view returns (int) {
            return _getCount();
        }

        function incrementCounter() public returns (int) {
            count += 1;
            return _getCount();
        }

        function decrementCounter() public returns (int) {
            count -= 1;
            return _getCount();
        }
    }
	"""
	print(SolcCompiler().compile(code, via_ir=False).hex())
	create_cfg(SolcCompiler().compile(code, via_ir=False))
	create_cfg(SolcCompiler().compile(code, via_ir=False), name="cfg_flatten.png", flatten=True)
