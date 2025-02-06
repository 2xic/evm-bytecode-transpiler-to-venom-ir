import solcx


class SolcCompiler:
	def __init__(self):
		pass
		
	def compile(self, file_content):
		output = self._get_solidity_output(file_content)
		return self._get_solc_bytecode(output, "main.sol")

	def _get_solidity_output(self, file_content):
		request = {
				"language": "Solidity",
				"sources": {
					"main.sol": {
						"content": file_content,
					}
				},
				"settings": {
					"outputSelection": {
						"*": {
							"*": [
								"evm.bytecode.object",
								"evm.deployedBytecode.object",
								"irOptimized",
							],
						}
					},
					"metadata":{
						"appendCBOR": False,
					},
				}
		}   
		return solcx.compile_standard(
			request,
			solc_binary="/usr/local/bin/solc-826"
		)

	def _get_solc_bytecode(self, output, file, key="deployedBytecode"):
		solc = output["contracts"][file]
		for i in list(solc.keys()):
			ref = solc[i]["evm"][key]["object"]
			code = bytes.fromhex(ref)
			if len(code) > 0:
				return code
		return None

