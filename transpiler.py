from opcodes import get_opcodes_from_bytes, PushOpcode
from blocks import get_basic_blocks
from symbolic import execute_block
from string import Template
import subprocess

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
		blocks = get_basic_blocks(get_opcodes_from_bytes(bytecode))
		for i in blocks:
			print(hex(i.start_offset))
			for v in i.opcodes:
				print("\t" + str(v))
		get_next_block = lambda idx: blocks[idx] if idx < len(blocks)  else None
		global_output = execute_block(blocks[0], get_next_block(1))
		for index, i in enumerate(global_output):
			global_output[index] = ("		" + i)

		new_blocks = []
		for index, block in enumerate(blocks[1:]):
			block_ir = execute_block(block, get_next_block(index+2))
			if len(block_ir) > 0:
				block_ir[0] = ("	" + block_ir[0])
				for index, i in enumerate(block_ir[1:]):
					block_ir[index + 1] = ("		" + i)
				new_blocks += block_ir
		# TODO: fallback for now, need to cleanup
		new_blocks.append("	" + "block_1337:")
		i = "revert 0,0"
		new_blocks.append("		" + i)
		
		#print("\n".join(output))
		return self.template.safe_substitute(
			global_template="\n".join(global_output),
			new_blocks="\n".join(new_blocks)
		)


def assert_compilation(bytecode):
	output = Transpiler().transpile(
		bytecode
	)
	print(output)
	with open("debug.venom", "w") as file:
		file.write(output)

	result = subprocess.run(["python3", "-m", "vyper.cli.venom_main", "debug.venom"], capture_output=True, text=True)
	print("STDOUT:", result.stdout)
	print("STDERR:", result.stderr)


if __name__ == "__main__":
	programs = [
		# https://www.evm.codes/playground?unit=Wei&codeType=Mnemonic&code=%27y1z0z0twwy2v32%200xsssszt%27~uuuuzv1%201y%2F%2F%20Example%20w%5CnvwPUSHuFFtwADDs~~%01stuvwyz~_
	#	bytes.fromhex("604260005260206000F3"),

		bytes.fromhex("6080604052348015600e575f80fd5b50600436106026575f3560e01c8063f8a8fd6d14602a575b5f80fd5b60306044565b604051603b91906062565b60405180910390f35b5f6001905090565b5f819050919050565b605c81604c565b82525050565b5f60208201905060735f8301846055565b9291505056")
	]
	for i in programs:
		assert_compilation(i)









