from opcodes import get_opcodes_from_bytes, PushOpcode
from blocks import get_basic_blocks
from ir_converter import execute_block
from string import Template
import subprocess
from blocks import get_calling_blocks
from test_utils.compiler import SolcCompiler
from cfg import create_cfg

class Transpiler:
	def __init__(self):
		self.template = Template("""
function global {
	global:
$global_template

$new_blocks
}
[data]
		""")
		self.variabl_counter = 0

	def transpile(self, bytecode):
		cfg = get_calling_blocks(get_opcodes_from_bytes(bytecode))
		for i in cfg.blocks:
			print(hex(i.start_offset))
			for v in i.opcodes:
				print("\t" + str(v))
		global_output = execute_block(cfg.blocks[0])
		for index, i in enumerate(global_output):
			global_output[index] = ("		" + i)

		new_blocks = []
		for index, block in enumerate(cfg.blocks[1:]):
			block_ir = execute_block(block)
			if len(block_ir) > 0:
				block_ir[0] = ("	" + block_ir[0])
				for index, i in enumerate(block_ir[1:]):
					block_ir[index + 1] = ("		" + i)
				new_blocks += block_ir
		return self.template.safe_substitute(
			global_template="\n".join(global_output),
			new_blocks="\n".join(new_blocks)
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


	result = subprocess.run(["python3", "-m", "vyper.cli.venom_main", "debug.venom"], capture_output=True, text=True)
	assert result.returncode == 0, result.stderr
	return bytes.fromhex(result.stdout.replace("0x", ""))

if __name__ == "__main__":
	code = """
    contract Hello {
        function test() public returns (uint256) {
            if (block.number < 15){
                return 2;
            }
            return 1;
        }
    }
	"""
	bytecode = SolcCompiler().compile(code, via_ir=False)
	venom_ir = Transpiler().transpile(
		bytecode
	)
	print(venom_ir)
#	print(venom_ir)
	create_cfg(compile_venom_ir(venom_ir))






