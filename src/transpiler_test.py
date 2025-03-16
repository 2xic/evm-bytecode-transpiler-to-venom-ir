from test_utils.compiler import SolcCompiler, OptimizerSettings
from transpiler import get_ssa_program
from test_utils.evm import execute_function
from test_utils.abi import encode_function_call
from eval import run_eval
from symbolic import EVM
import subprocess

def execute_evm(bytecode_a, bytecode_b, function):
	out_a = execute_function(bytecode_a, function)
	out_b = execute_function(bytecode_b, function)
	assert out_a == \
			out_b, f"{out_a} != {out_b} with {function.hex()}"
	return True


def compile_venom_ir(output):
	with open("debug.venom", "w") as file:
		file.write(output)

	# TODO: import it as module instead ? 
	result = subprocess.run(["python3", "-m", "vyper.cli.venom_main", "debug.venom"], capture_output=True, text=True)
	assert result.returncode == 0, result.stderr
	return bytes.fromhex(result.stdout.replace("0x", ""))


def test_simple_hello_world():
	code = """
	contract Hello {
		function test() public returns (uint256) {
			return 1;
		}
	}
	"""
	output = SolcCompiler().compile(code)
	transpiled = compile_venom_ir(get_ssa_program(output).process().convert_into_vyper_ir())
	
	assert execute_evm(
		output,
		transpiled,
		encode_function_call("test()"),        
	)
	assert len(transpiled) < len(output)

def test_simple_multiple_functions():
	code = """
	contract Hello {
		function test() public returns (uint256) {
			return bagel();
		}

		function bagel() public returns (uint256) {
			return 1;
		}
	}
	"""
	output = SolcCompiler().compile(code)
	transpiled = compile_venom_ir(get_ssa_program(output).process().convert_into_vyper_ir())
	
	assert execute_evm(
		output,
		transpiled,
		encode_function_call("bagel()"),        
	)
	# TODO: Figure out what the issue is.
	assert execute_evm(
		output,
		transpiled,
		encode_function_call("test()"),        
	)
	assert execute_evm(
		output,
		transpiled,
		encode_function_call("fest()"),        
	)

def test_should_handle_loops():
	code = """
	contract Counter {
		int private count = 0;

		function _getCount() internal view returns (int) {
			return count;
		}

		function getCount() public view returns (int) {
			return _getCount();
		}

		function incrementCounter() public returns (int) {
			count += 1;
			return _getCount();
		}

		function decrementCounter() public returns (int) {
			count -= 1;
			return _getCount();
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	# TODO: add compilation check here also.	

def test_should_handle_phi_djmps():
	code = """
	contract Hello {
		function test() public returns (uint256) {
			return test2();
		}

		function test2() public returns (uint256) {
			return 1;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom_ir(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),        
	)
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test2()"),        
	)

def test_should_handle_storage():
	code = """
	contract Hello {
		uint256 public val = 0;

		function set() public returns (uint) {
			val = 50;
			return val;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False

	transpiled = compile_venom_ir(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("set()"),        
	)

# TODO: need to handle cycles in the CFG.
def test_should_handle_control_flow():
	code = """
	contract Hello {
		function sumUpTo() public pure returns (uint) {
			uint sum = 0;
			for (uint i = 1; i <= 10; i++) {
				sum += i;
			}
			return sum;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code, OptimizerSettings().optimize(
		optimization_runs=2 ** 31 - 1
	))
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom_ir(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("sumUpTo()"),        
	)

def skip_test_should_handle_control_flow():
	code = """
	contract Hello {
		function sumUpTo() public pure returns (uint) {
			uint sum = 0;
			for (uint i = 1; i <= 10; i++) {
				sum += i;
			}
			return sum;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code, OptimizerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	"""
	transpiled = compile_venom_ir(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("sumUpTo()"),        
	)
	"""
def test_nested_if_conditions_explicit():
	code = """
	contract Hello {
		function test() public returns (uint256) {
			if (5 < 10) {        
				return 2;
			}
			return 0;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom_ir(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)

def test_nested_if_conditions_params_explicit():
	code = """
	contract Hello {
		function test() public returns (uint256) {
			return a(15);
		}

		function a(uint256 a) internal returns (uint256){
			if (a > 10) {        
				return 2;
			}
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom_ir(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)

def test_block_conditions():
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
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom_ir(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)


def skip_test_simple_mapping_no_optimizations():
	code = """
	contract Hello {
		mapping(uint256 => uint256) public value;

		function test() public returns (uint256) {
			value[0] = 1;
			return value[0];
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom_ir(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)

def test_simple_mapping():
	code = """
	contract Hello {
		mapping(uint256 => uint256) public value;

		function test() public returns (uint256) {
			value[0] = 1;
			return value[0];
		}
	}
	"""
	bytecode = SolcCompiler().compile(code, OptimizerSettings().optimize())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom_ir(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("test()"),
	)

def test_multiple_functions():
	code = """
	contract Hello {
		function timestamp() public returns (uint256) {
			return block.timestamp;
		}

		function number() public returns (uint256) {
			return block.number;
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	transpiled = compile_venom_ir(output.convert_into_vyper_ir())
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("timestamp()"),
	)
	assert execute_evm(
		bytecode,
		transpiled,
		encode_function_call("number()"),
	)

def test_should_handle_coin_example():
	code = """
	// From https://docs.soliditylang.org/en/latest/introduction-to-smart-contracts.html#subcurrency-example
	contract Coin {
		// The keyword "public" makes variables
		// accessible from other contracts
		address public minter;
		mapping (address => uint) public balances;

		// Events allow clients to react to specific
		// contract changes you declare
		event Sent(address from, address to, uint amount);

		// Constructor code is only run when the contract
		// is created
		constructor() public {
			minter = msg.sender;
		}

		// Sends an amount of newly created coins to an address
		// Can only be called by the contract creator
		function mint(address receiver, uint amount) public {
			require(msg.sender == minter);
			require(amount < 1e60);
			balances[receiver] += amount;
		}

		// Sends an amount of existing coins
		// from any caller to an address
		function send(address receiver, uint amount) public {
			require(amount <= balances[msg.sender], "Insufficient balance.");
			balances[msg.sender] -= amount;
			balances[receiver] += amount;
			emit Sent(msg.sender, receiver, amount);
		}
	}
	"""
	bytecode = SolcCompiler().compile(code, OptimizerSettings())
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	if False:
		transpiled = compile_venom_ir(output.convert_into_vyper_ir())

def test_should_handle_sstore():
	code = """
	// From https://docs.soliditylang.org/en/latest/introduction-to-smart-contracts.html#subcurrency-example
	contract SimpleMapping {
		mapping(address => mapping(address => bool)) public mappings;

		function setResults(address value) public returns(address) {
			mappings[address(0)][value] = true;
			return value;
		}

		function getResults(address value) public returns (bool) {
			return mappings[address(0)][value];
		}
	}
	"""
	bytecode = SolcCompiler().compile(code)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
	if False:
		transpiled = compile_venom_ir(output.convert_into_vyper_ir())
		assert execute_evm(
			bytecode,
			transpiled,
			encode_function_call("setResults(address)", ["address"], ['0x' + 'ff' * 20]),
		)

# Sanity check that the eval script should still work
def test_eval():
	run_eval()

def test_stack_evm():
	evm = EVM(0)
	evm.stack.append(1)
	evm.stack.append(2)
	evm.swap(1)
	assert evm.stack == [2, 1]
	evm.stack.append(3)
	evm.swap(1)
	assert evm.stack == [2, 3, 1]

	evm.stack = [2, 0, 0, 1]
	evm.swap(3)
	assert evm.stack == [1, 0, 0, 2]

	evm.stack = [1, 0, 0]
	evm.dup(3)
	assert evm.stack == [1, 0, 0, 1]

