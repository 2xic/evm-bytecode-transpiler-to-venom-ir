from test_utils.compiler import SolcCompiler, OptimizerSettings
from opcodes import get_opcodes_from_bytes
from blocks import get_calling_blocks
import graphviz
import argparse

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

def cfg_from_single_solidity_file(filepath, via_ir):
	optimization_settings = OptimizerSettings().optimize(
		optimization_runs=2 ** 31 - 1
	) if via_ir else OptimizerSettings()
	optimization_settings.deduplicate = True
	with open(filepath, "r") as file:
		code = file.read()
		bytecode = SolcCompiler().compile(code, settings=optimization_settings)
		return cfg_from_bytecode(bytecode)

def cfg_from_bytecode(bytecode):
	save_cfg(bytecode)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# input source
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--filepath', type=str, help='Path to the file')
	group.add_argument('--bytecode', type=str, help='Bytecode as a hex string')

	# options
	parser.add_argument("--via-ir", default=False, action='store_true')

	args = parser.parse_args()

	if args.filepath:
		cfg_from_single_solidity_file(args.filepath, args.via_ir)
	elif args.bytecode:
		bytecode = bytes.fromhex(args.bytecode.replace("0x",""))
		cfg_from_bytecode(bytecode)
