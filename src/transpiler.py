from opcodes import get_opcodes_from_bytes
from ir_converter import execute_block
from ir_optimizer import optimize_ir
from ir_phi_handling import PhiHelperUtil
from string import Template
import subprocess
from blocks import get_calling_blocks
from test_utils.compiler import SolcCompiler
from cfg import create_cfg
from test_utils.evm import execute_function
from test_utils.abi import encode_function_call

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

		phi_functions = PhiHelperUtil().get_opcodes_assignments(cfg.blocks, cfg.blocks_lookup)
		blocks = []
		global_variables = {}
		for index, block in enumerate(cfg.blocks):
			blocks.append(execute_block(block, get_next_block(index + 1), global_variables, phi_functions))
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
            return a(15);
        }

        function fest() public returns (uint256) {
            return a(5);
        }

        function a(uint256 a) internal returns (uint256){
            if (a > 10) {        
                return 2;
            }
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
	for name, code in zip(["solidity", "venom"], [bytecode, venom_bytecode]):
		print(name)
		print(execute_function(
			code,
			encode_function_call("test()")
		))
		print(execute_function(
			code,
			encode_function_call("fest()")
		))

