import solcx
from dataclasses import dataclass
import cbor2


def strip_metadata(bytecode: bytes):
	metadata_length = int.from_bytes(bytecode[-2:], byteorder="big")

	if metadata_length > 0 and metadata_length <= len(bytecode):
		metadata_end = len(bytecode) - 2
		metadata_start = metadata_end - (metadata_length - 2)
		if metadata_start < 0:
			return bytecode

		potential_metadata = bytecode[metadata_start:metadata_end]

		try:
			cbor2.loads(potential_metadata)
			return bytecode[: len(bytecode) - metadata_length]
		except (cbor2.CBORDecodeError, ValueError):
			pass
	return bytecode


@dataclass
class CompilerSettings:
	via_ir: bool = False
	optimizer_enabled: bool = False
	optimization_runs: int = 200
	solc_version: str = "0.8.29"

	def optimize(self, optimization_runs=200, via_ir=True):
		self.via_ir = via_ir
		self.optimization_runs = optimization_runs
		self.optimizer_enabled = True
		return self


class SolcCompiler:
	def compile(self, file_content, settings=CompilerSettings()) -> bytes:
		output = self._get_solidity_output(file_content, settings)
		return self._get_solc_bytecode(output, "main.sol")

	def compile_yul(self, file_content, settings=CompilerSettings()) -> bytes:
		solcx.install_solc(settings.solc_version)
		request = self._get_standard_json(file_content, target="Yul")
		return self._get_solc_bytecode(
			solcx.compile_standard(
				request,
				solc_version=settings.solc_version,
			),
			"main.sol",
			key="bytecode",
		)

	def _get_solidity_output(self, file_content, settings: CompilerSettings):
		solcx.install_solc(settings.solc_version)
		request = self._get_standard_json(
			file_content,
			settings,
		)
		return solcx.compile_standard(
			request,
			solc_version=settings.solc_version,
		)

	def _get_standard_json(
		self,
		file_content: str,
		settings: CompilerSettings = CompilerSettings(),
		target="Solidity",
	):
		return {
			"language": target,
			"sources": {
				"main.sol": {
					"content": file_content,
				}
			},
			"settings": {
				"outputSelection": {
					"*": {
						"*": [
							# Used by the yul compilation
							"evm.bytecode.object",
							"evm.deployedBytecode.object",
							"irOptimized",
						],
					}
				},
				"optimizer": {
					"enabled": settings.optimizer_enabled,
					"runs": settings.optimization_runs,
					#"details": {
					#	"deduplicate": False,
					#	"constantOptimizer": False,
					#	"orderLiterals": False,
					#	"cse": False,
					#	"inliner": False,
					#}
				},
				"viaIR": settings.via_ir,
			},
		}

	def _get_solc_bytecode(self, output, file, key="deployedBytecode"):
		solc = output["contracts"][file]
		for i in list(solc.keys()):
			ref = solc[i]["evm"][key]["object"]
			code = bytes.fromhex(ref)
			if len(code) > 0:
				return strip_metadata(code)
		return None
