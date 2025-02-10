from opcodes import get_opcodes_from_bytes
from ir_converter import execute_block, optimize_ir, create_missing_blocks
from string import Template
import subprocess
from blocks import get_calling_blocks
from test_utils.compiler import SolcCompiler
from cfg import create_cfg

class Transpiler:
	def __init__(self):
		self.template = Template("""
function global {
$blocks
}
[data]
		""")

	def transpile(self, bytecode):
		cfg = get_calling_blocks(get_opcodes_from_bytes(bytecode))
		get_next_block = lambda idx: cfg.blocks[idx] if idx < len(cfg.blocks)  else None
		blocks = []
		for index, block in enumerate(cfg.blocks):
			blocks += execute_block(block, get_next_block(index + 1), cfg.blocks_lookup)
		blocks = create_missing_blocks(blocks)
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
            return block.timestamp;
        }

        function fest() public returns (uint256) {
            return block.number;
        }
    }
	"""
	bytecode = SolcCompiler().compile(code, via_ir=False)
	create_cfg(bytecode, "solc.png")
	venom_ir = Transpiler().transpile(
		bytecode
	)
	print(venom_ir)
#	print(venom_ir)
	create_cfg(compile_venom_ir(venom_ir), "venom.png")






