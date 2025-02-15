from opcodes import get_opcodes_from_bytes
from ir_converter import execute_block, optimize_ir, create_missing_blocks, InsertPhiFunction, delta_executions
from string import Template
import subprocess
from blocks import get_calling_blocks
from test_utils.compiler import SolcCompiler
from cfg import create_cfg
from test_utils.evm import execute_function
from test_utils.abi import encode_function_call
from typing import List

class Transpiler:
	def __init__(self):
		self.template = Template("""
function global {
$blocks
}
""")

	def transpile(self, bytecode):
		cfg = get_calling_blocks(get_opcodes_from_bytes(bytecode))
		get_next_block = lambda idx: cfg.blocks[idx] if idx < len(cfg.blocks)  else None

		phi_functions: List[InsertPhiFunction] = []
		for index, block in enumerate(cfg.blocks):
			_, out = delta_executions(cfg.blocks_lookup, block.execution_trace)
			for v in out:
				phi_functions.append(v)		

		blocks = []
		for index, block in enumerate(cfg.blocks):
			blocks += execute_block(block, get_next_block(index + 1), cfg.blocks_lookup, phi_functions)
		# blocks = create_missing_blocks(blocks)
		blocks = optimize_ir(blocks)
		return self.template.safe_substitute(
			blocks="\n".join(list(map(str, blocks)))
		)

def assert_compilation(bytecode):
	output = Transpiler().transpile(
		bytecode
	)
	print(output)
	return compile_venom_ir(output)

def compile_venom_ir(output):
	with open("debug.venom", "w") as file:
		file.write(output)

	# TODO: import it as module instead ? 
	result = subprocess.run(["python3", "-m", "vyper.cli.venom_main", "debug.venom"], capture_output=True, text=True)
	assert result.returncode == 0, result.stderr
	return bytes.fromhex(result.stdout.replace("0x", ""))

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
	bytecode = SolcCompiler().compile(code, via_ir=False)
	create_cfg(bytecode, "solc.png")
	venom_ir = Transpiler().transpile(
		bytecode
	)
	venom_bytecode = compile_venom_ir(venom_ir)
	print(encode_function_call("test()").hex())
	print(venom_bytecode.hex())

	create_cfg(venom_bytecode, "venom.png")
	print("OUT:")
	print(execute_function(
		venom_bytecode,
		encode_function_call("test()")
	))
	print(execute_function(
		venom_bytecode,
		encode_function_call("bagel()")
	))

