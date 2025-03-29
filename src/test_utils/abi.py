import sha3
from eth_abi import encode


def keccak(value: bytes) -> bytes:
	k = sha3.keccak_256()
	k.update(value)
	return k.digest()


def encode_arguments(types, values):
	assert len(types) == len(values)
	encoded = encode(types, values) if len(types) > 0 else b""
	return encoded


def encode_function_call(function_name: str, types=[], values=[]):
	assert "(" in function_name and ")" in function_name, "missing function signature"
	encoded = encode_arguments(types, values)
	sighash = keccak(function_name.encode())[:4]
	return bytes(4 - len(sighash)) + (sighash + encoded)
