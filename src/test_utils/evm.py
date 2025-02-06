from eth.vm.forks.cancun import CancunVM
from eth.db.atomic import AtomicDB
from eth.chains.base import MiningChain
from eth.rlp.headers import BlockHeader
from eth.abc import ComputationAPI

CODE_ADDRESS = b'\x21' * 20
SENDER_ADDRESS = b'\x22' * 20

class SimpleChain(MiningChain):
	vm_configuration = ((0, CancunVM),) 

def run_vm(bytecode: bytes, data: bytes, storage={}, wei_value=0) -> ComputationAPI:
	chain_db = AtomicDB()
	genesis_header = BlockHeader(difficulty=1, block_number=0, gas_limit=21_000)
	chain = SimpleChain.from_genesis_header(chain_db, genesis_header)

	# Use the VM to apply the message (this runs the bytecode)

	vm = chain.get_vm()

	for key, value in storage.items():
		vm.state.set_storage(CODE_ADDRESS, key, value)
	vm.state.set_code(CODE_ADDRESS, bytecode)
	vm.state.set_balance(SENDER_ADDRESS, 20000000000)
	vm.state.set_balance(CODE_ADDRESS, 	 20000000000)

	computation = vm.execute_bytecode(
		gas_price=1,
		origin=SENDER_ADDRESS,
		to=CODE_ADDRESS,
		sender=SENDER_ADDRESS,
		value=wei_value,
		gas=1000000,
		code_address=CODE_ADDRESS,
		data=data,
		code=bytecode,
	)
	return computation
