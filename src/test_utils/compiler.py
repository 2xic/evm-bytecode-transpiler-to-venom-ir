import solcx
from dataclasses import dataclass

@dataclass
class OptimizerSettings:
	via_ir: bool = False
	optimizer_enabled: bool = False
	optimization_runs: int = 200
	deduplicate: bool = False
	evm_version: str = "paris"
	solc_version: str = "0.8.26"

	def optimize(self, optimization_runs = 200, via_ir=True):
		self.via_ir = via_ir
		self.optimization_runs = optimization_runs
		self.optimizer_enabled = True
		return self 

class SolcCompiler:
	def compile(self, file_content, settings = OptimizerSettings()) -> bytes:
		output = self._get_solidity_output(file_content, settings)
		return self._get_solc_bytecode(output, "main.sol")

	def _get_solidity_output(self, file_content, settings: OptimizerSettings):
		solcx.install_solc(settings.solc_version)
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
					"evmVersion": settings.evm_version,
					"optimizer": {
						"enabled": settings.optimizer_enabled,
						"runs": settings.optimization_runs,
						"details": {
							# Causes a lot of phi function to be needed
							"deduplicate": False,
						},
					},
					"viaIR": settings.via_ir,
				}
		}   
		return solcx.compile_standard(
			request,
			solc_version=settings.solc_version,
		)

	def _get_solc_bytecode(self, output, file, key="deployedBytecode"):
		solc = output["contracts"][file]
		for i in list(solc.keys()):
			ref = solc[i]["evm"][key]["object"]
			code = bytes.fromhex(ref)
			if len(code) > 0:
				return code
		return None
