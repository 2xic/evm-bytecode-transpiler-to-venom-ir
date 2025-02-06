from opcodes import get_opcodes_from_bytes, PushOpcode
from string import Template
import subprocess

class Transpiler:
	def __init__(self):
		self.template = Template("""
function global {
	global:
$global_template
}
		""")
		self.variabl_counter = 0

	def transpile(self, bytecode):
		output = []
		for i in get_opcodes_from_bytes(bytecode):
			if isinstance(i, PushOpcode):
				val = i.data.hex()
				output.append(f"%{self.variabl_counter} = {val}")
				self.variabl_counter += 1
			elif i.name == "MSTORE":
				output.append(f"mstore %{self.variabl_counter - 2}, %{self.variabl_counter-1}")
				self.variabl_counter += 1
			elif i.name == "RETURN":
				output.append(f"return %{self.variabl_counter - 2}, %{self.variabl_counter-1}")
				self.variabl_counter += 1
			else:
				print("Unknonw ", i.name)

		for index, i in enumerate(output):
			output[index] = ("		" + i)

		#print("\n".join(output))
		return self.template.safe_substitute(
			global_template="\n".join(output),
		)

if __name__ == "__main__":
	output = Transpiler().transpile(
		bytes.fromhex("604260005260206000F3")
	)
	with open("debug.venom", "w") as file:
		file.write(output)
	result = subprocess.run(["python3", "-m", "vyper.cli.venom_main", "debug.venom"], capture_output=True, text=True)
	print("STDOUT:", result.stdout)
	print("STDERR:", result.stderr)

