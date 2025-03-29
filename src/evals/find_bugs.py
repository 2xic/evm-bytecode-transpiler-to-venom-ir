"""
Fetches deployed contracts from a Dune Query and then tries to transpile it.
"""

import os
import requests
from dotenv import load_dotenv
from bytecode_transpiler.transpiler import transpile_from_bytecode
import json
import time
import cbor2

load_dotenv()

ETHERSCAN_API_KEY = os.environ["ETHERSCAN_API_KEY"]
DUNE_API_KEY = os.environ["DUNE_API_KEY"]
query_id = os.environ["QUERY_ID"]


class Cache:
	def get(self, name):
		path = self.path(name)
		if os.path.isfile(path):
			with open(path, "r") as file:
				return json.load(file)
		return None

	def put(self, name, response):
		path = self.path(name)
		print(path)
		with open(path, "w") as file:
			file.write(json.dumps(response, indent=4))

	def path(self, name):
		return os.path.join(os.path.dirname(__file__), "../", ".cache", name)


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


cache = Cache()


def do_compile(code):
	try:
		result = transpile_from_bytecode(code)
		return {
			"result": result,
		}
	except Exception as e:
		return {"error": str(e)}


def get_compiler_info(address):
	key = f"etherscan_{address}"
	if (data := cache.get(key)) is None:
		data = requests.get(
			f"https://api.etherscan.io/api?module=contract&action=getsourcecode&address={address}&apikey={ETHERSCAN_API_KEY}"
		).json()
		cache.put(key, data)
		time.sleep(3)
	# TODO, this will not correctly detect via ir etc.
	if (
		data["status"] == "1"
		and data["result"][0]["ABI"] != "Contract source code not verified"
	):
		return {
			"CompilerVersion": data["result"][0]["CompilerVersion"],
			"OptimizationUsed": data["result"][0]["OptimizationUsed"],
			"Proxy": data["result"][0]["Proxy"],
		}


def main():
	base_url = f"https://api.dune.com/api/v1/query/{query_id}/results"
	headers = {"X-Dune-API-Key": DUNE_API_KEY}
	result_response = requests.request("GET", base_url, headers=headers).json()

	print("Loaded dataset")

	errors = 0
	success = 0
	total_count_fail = 0
	error_bytecode = []

	for i in result_response["result"]["rows"]:
		code = strip_metadata(bytes.fromhex(i["code"].lstrip("0x")))
		if len(code) > 0:
			print(i["name"], len(code))

			return_dict = do_compile(code)
			if "error" in return_dict:
				errors += 1
				total_count_fail += 1
				error_bytecode.append(
					{
						"bytecode": code.hex(),
						"error": return_dict["error"],
					}
				)
				with open("errors.json", "w") as file:
					file.write(json.dumps(error_bytecode, indent=4))
			elif "result" in return_dict:
				output = return_dict["result"]
				print(len(code) / len(output))
				success += 1
				total_count_fail = 0
			else:
				errors += 1
				total_count_fail += 1

			print((errors, success, (success) / (errors + success)))
			if total_count_fail > 15:
				break

	print((success) / (errors + success))


if __name__ == "__main__":
	main()
