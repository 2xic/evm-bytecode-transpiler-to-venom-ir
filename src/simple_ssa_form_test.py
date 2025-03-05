from test_utils.compiler import SolcCompiler
from simpler_ssa_form import get_ssa_program
from test_utils.compiler import SolcCompiler
from test_utils.evm import execute_function
from test_utils.abi import encode_function_call
from transpiler import assert_compilation
from symbolic import EVM
import subprocess

def execute_evm(bytecode_a, bytecode_b, function):
	out_a = execute_function(bytecode_a, function)
	out_b = execute_function(bytecode_b, function)
   # #print((out_a, out_b))
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
	bytecode = SolcCompiler().compile(code, via_ir=False)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False

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
	bytecode = SolcCompiler().compile(code, via_ir=False)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False

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
	bytecode = SolcCompiler().compile(code, via_ir=False)
	output = get_ssa_program(bytecode)
	output.process()
	assert output.has_unresolved_blocks == False
