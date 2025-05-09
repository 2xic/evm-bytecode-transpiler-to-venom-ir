from v2.symbolic_execution_state import (
	ProgramExecution,
	SsaProgramBuilder,
)
from test_utils.bytecodes import SINGLE_BLOCK, ERC721_DROP, MINIMAL_PROXY, MINIMAL_PROXY_2
from test_utils.solc_compiler import SolcCompiler, CompilerSettings
import pytest


def normalize(x: str):
	return "\n".join(list(map(lambda x: x.strip(), filter(lambda x: len(x) > 0, x.strip().split("\n")))))


def are_equal_ignoring_spaces(str1, str2):
	return normalize(str1) == normalize(str2)


def test_basic_interface():
	program = ProgramExecution.create_from_bytecode(SINGLE_BLOCK)
	assert len(program.blocks) == 1
	assert len(program.execution) == 8

	program = ProgramExecution.create_from_bytecode(ERC721_DROP)
	assert len(program.blocks) == 44
	assert len(program.execution) == 61


def test_basic_ssa_program():
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(SINGLE_BLOCK),
	)
	output_block = program.create_program()
	assert are_equal_ignoring_spaces(
		"""
		global:
				%0 = CALLDATASIZE 
				%1 = RETURNDATASIZE 
				%2 = RETURNDATASIZE 
				CALLDATACOPY %2, %1, %0
				%4 = CALLDATASIZE 
				%5 = RETURNDATASIZE 
				%6 = CALLVALUE 
				%7 = CREATE %6, %5, %4
				STOP
		""",
		str(output_block),
	)
	code = output_block.compile()
	assert code.hex() == "363d3d37363d34f05000"


def test_minimal_proxy_program():
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(MINIMAL_PROXY),
	)
	output_block = program.create_program()
	assert are_equal_ignoring_spaces(
		"""
			global:
					%0 = CALLDATASIZE 
					%1 = RETURNDATASIZE 
					%2 = RETURNDATASIZE 
					CALLDATACOPY %2, %1, %0
					%4 = RETURNDATASIZE 
					%5 = RETURNDATASIZE 
					%6 = RETURNDATASIZE 
					%7 = CALLDATASIZE 
					%8 = RETURNDATASIZE 
					%10 = GAS 
					%11 = DELEGATECALL %10, 1097817159418366163791829159214798623611012571465, %8, %7, %6, %5
					%12 = RETURNDATASIZE 
					RETURNDATACOPY %4, %4, %12
					%17 = RETURNDATASIZE 
					JNZ %11, @block_0x2b, @block_0x2a
			block_0x2a:
					REVERT %4, %17
			block_0x2b:
					RETURN %4, %17
		""",
		str(output_block),
	)
	code = output_block.compile()
	assert (
		code.hex() == "363d3d373d3d3d90369073c04bd2f0d484b7e0156b21c98b2923ca8b9ce1495af43d3d9081823e3d9161002e57fd5bf3"
	)


def test_basic_add_program():
	# https://www.evm.codes/playground?unit=Wei&codeType=Mnemonic&code=%27y1z0z0twwy2v32%200xsssszt%27~uuuuzv1%201y%2F%2F%20Example%20w%5CnvwPUSHuFFtwADDs~~%01stuvwyz~_
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(bytes.fromhex("604260005260206000F3")),
	)
	output_block = program.create_program()
	print(str(output_block))
	assert are_equal_ignoring_spaces(
		"""
		global:
			MSTORE 0, 66
			RETURN 0, 32
		""",
		str(output_block),
	)
	code = output_block.compile()
	assert code.hex() == "60425f5260205ff3"


def test_basic_blocks():
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(
			bytes.fromhex("363d3d373d3d3d363d73c04bd2f0d484b7e0156b21c98b2923ca8b9ce1495af43d82803e903d91602b57fd5bf3")
		),
	)
	output_block = program.create_program()
	print(str(output_block))
	assert are_equal_ignoring_spaces(
		"""
			global:
					%0 = CALLDATASIZE 
					%1 = RETURNDATASIZE 
					%2 = RETURNDATASIZE 
					CALLDATACOPY %2, %1, %0
					%4 = RETURNDATASIZE 
					%5 = RETURNDATASIZE 
					%6 = RETURNDATASIZE 
					%7 = CALLDATASIZE 
					%8 = RETURNDATASIZE 
					%10 = GAS 
					%11 = DELEGATECALL %10, 1097817159418366163791829159214798623611012571465, %8, %7, %6, %5
					%12 = RETURNDATASIZE 
					RETURNDATACOPY %4, %4, %12
					%17 = RETURNDATASIZE 
					JNZ %11, @block_0x2b, @block_0x2a
			block_0x2a:
					REVERT %4, %17
			block_0x2b:
					RETURN %4, %17
		""",
		str(output_block),
	)
	code = output_block.compile()
	assert (
		code.hex() == "363d3d373d3d3d90369073c04bd2f0d484b7e0156b21c98b2923ca8b9ce1495af43d3d9081823e3d9161002e57fd5bf3"
	), code.hex()


def test_basic_for_loop():
	yul_code = """
		object "Loop" {
			code {
				let i := 0
				for { } lt(i, 10) { i := add(i, 1) } {
					mstore(add(0x00, mul(i, 0x20)), i)
				}
				return(0x00, 0x200)
			}
		}
	"""
	code = SolcCompiler().compile_yul(yul_code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	print(str(output_block))
	assert are_equal_ignoring_spaces(
		"""
			global:
					%0 = 0
					JMP @block_0x1
			block_0x1:
					%phi0 = phi @global, %0, @block_0xe, %20
					%4 = LT %phi0, 10
					JNZ %4, @block_0xe, @block_0x9
			block_0x9:
					RETURN 0, 512
			block_0xe:
					%16 = MUL %phi0, 32
					%18 = ADD 0, %16
					MSTORE %18, %phi0
					%20 = ADD %phi0, 1
					JMP @block_0x1
		""",
		str(output_block),
	)
	assert are_equal_ignoring_spaces(
		"""
			function global {
					global:
							%0 = 0
							jmp @block_0x1
					block_0x1:
							%phi0 = phi @global, %0, @block_0xe, %20
							%4 = lt %phi0, 10
							jnz %4, @block_0xe, @block_0x9
					block_0x9:
							return 0, 512
					block_0xe:
							%16 = mul %phi0, 32
							%18 = add 0, %16
							mstore %18, %phi0
							%20 = add %phi0, 1
							jmp @block_0x1
			}
		""",
		output_block.convert_to_vyper_ir(),
	)
	assert isinstance(output_block.compile(), bytes)


def test_dynamic_jump():
	yul_code = """
		{			
			switch selector()
			case 0x0dbe671f {
				let result := a()
				returnUint(result)
			}
			case 0x4df7e3d0 {
				let result := b()
				returnUint(result)
			}
			default {
				revert(0, 0)
			}
			
			function a() -> result {
				result := c(10)
				result := add(result, 5)
			}
			
			function b() -> result {
				result := c(20) 
				result := mul(result, 2)
			}
			
			function c(x) -> result {
				result := add(mul(x, 2), 1)
			}
			
			function selector() -> s {
				s := div(calldataload(0), 0x100000000000000000000000000000000000000000000000000000000)
			}
			
			function returnUint(v) {
				mstore(0, v)
				return(0, 32)
			}
		}
	"""
	code = SolcCompiler().compile_yul(yul_code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	print(output_block)
	assert are_equal_ignoring_spaces(
		"""
			global:
					JMP @block_0x56
			block_0x5:
					%6 = EQ 230582047, %64
					JNZ %6, @block_0x26, @block_0x10
			block_0x10:
					%10 = EQ 1308091344, %64
					JNZ %10, @block_0x1c, @block_0x19
			block_0x19:
					REVERT 0, 0
			block_0x1c:
					JMP @block_0x3e
			block_0x22:
					JMP @block_0x61
			block_0x26:
					JMP @block_0x30
			block_0x2c:
					JMP @block_0x61
			block_0x30:
					%32 = @block_0x3a
					%33 = 10
					JMP @block_0x4c
			block_0x3a:
					%37 = ADD %55, 5
					JMP @block_0x2c
			block_0x3e:
					%42 = @block_0x48
					%43 = 20
					JMP @block_0x4c
			block_0x48:
					%47 = MUL %55, 2
					JMP @block_0x22
			block_0x4c:
					%phi8 = phi @block_0x30, %33, @block_0x3e, %43
					%phi10 = phi @block_0x30, %32, @block_0x3e, %42
					%54 = MUL %phi8, 2
					%55 = ADD %54, 1
					DJMP %phi10, @block_0x3a, @block_0x48
			block_0x56:
					%61 = SHL 224, 1
					%63 = CALLDATALOAD 0
					%64 = DIV %63, %61
					JMP @block_0x5
			block_0x61:
					%phi12 = phi @block_0x2c, %37, @block_0x22, %47
					MSTORE 0, %phi12
					RETURN 0, 32
		""",
		str(output_block),
	)
	assert isinstance(output_block.compile(), bytes)


@pytest.mark.parametrize(
	"raw_bytecode",
	[
		MINIMAL_PROXY.hex(),
		MINIMAL_PROXY_2.hex(),
	],
)
def test_raw_bytecode(raw_bytecode):
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(bytes.fromhex(raw_bytecode)),
	)
	output_block = program.create_program()
	code = output_block.compile()
	assert isinstance(code, bytes)


@pytest.mark.parametrize(
	"code",
	[
		"""
		contract Hello {
			function test() public returns (uint256) {
				return 1;
			}
		}
		""",
	],
)
def test_hello_solidity(code):
	code = SolcCompiler().compile(code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	code = output_block.compile()
	assert isinstance(code, bytes)


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
	code = SolcCompiler().compile(code)
	print(code.hex())
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	code = output_block.compile()
	assert isinstance(code, bytes)


def test_counter():
	# From https://solidity-by-example.org/first-app/
	code = """
		contract Counter {
			uint256 public count;

			// Function to get the current count
			function get() public view returns (uint256) {
				return count;
			}

			// Function to increment count by 1
			function inc() public {
				count += 1;
			}

			// Function to decrement count by 1
			function dec() public {
				// This function will fail if count = 0
				count -= 1;
			}
		}
	"""
	code = SolcCompiler().compile(
		code,
	)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	code = output_block.compile()
	assert isinstance(code, bytes)


def test_send_eth():
	# From https://solidity-by-example.org/sending-ether/
	code = """
	contract SendEther {
		function sendViaTransfer(address payable _to) public payable {
			// This function is no longer recommended for sending Ether.
			_to.transfer(msg.value);
		}

		function sendViaSend(address payable _to) public payable {
			// Send returns a boolean value indicating success or failure.
			// This function is not recommended for sending Ether.
			bool sent = _to.send(msg.value);
			require(sent, "Failed to send Ether");
		}

		function sendViaCall(address payable _to) public payable {
			// Call returns a boolean value indicating success or failure.
			// This is the current recommended method to use.
			(bool sent, bytes memory data) = _to.call{value: msg.value}("");
			require(sent, "Failed to send Ether");
		}
	}
	"""
	code = SolcCompiler().compile(
		code,
	)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	print(output_block.convert_to_vyper_ir())
	code = output_block.compile()
	assert isinstance(code, bytes)


def test_assembly_variable():
	# From https://solidity-by-example.org/assembly-variable/
	code = """
		contract AssemblyVariable {
			function yul_let() public pure returns (uint256 z) {
				assembly {
					// The language used for assembly is called Yul
					// Local variables
					let x := 123
					z := 456
				}
			}
		}
	"""
	code = SolcCompiler().compile(
		code,
	)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	code = output_block.compile()
	assert isinstance(code, bytes)


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
	code = SolcCompiler().compile(code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	code = output_block.compile()
	assert isinstance(code, bytes)


def test_loop():
	# From https://solidity-by-example.org/loop/
	code = """
		contract Loop {
			function loop() public pure {
				// for loop
				for (uint256 i = 0; i < 10; i++) {
					if (i == 3) {
						// Skip to next iteration with continue
						continue;
					}
					if (i == 5) {
						// Exit loop with break
						break;
					}
				}

				// while loop
				uint256 j;
				while (j < 10) {
					j++;
				}
			}
		}
	"""
	code = SolcCompiler().compile(code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	code = output_block.compile()
	assert isinstance(code, bytes)


def test_constants():
	# From https://solidity-by-example.org/constants/
	code = """
	contract Constants {
		// coding convention to uppercase constant variables
		address public constant MY_ADDRESS =
			0x777788889999AaAAbBbbCcccddDdeeeEfFFfCcCc;
		uint256 public constant MY_UINT = 123;

		function test() public returns (uint256) {
			return MY_UINT;
		}
	}
	"""
	code = SolcCompiler().compile(code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	code = output_block.compile()
	assert isinstance(code, bytes)


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
	code = SolcCompiler().compile(code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	code = output_block.compile()
	assert isinstance(code, bytes)


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
	code = SolcCompiler().compile(code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program()
	code = output_block.compile()
	assert isinstance(code, bytes)

def test_v2_simple_mapping():
	code = """
	contract Hello {
		mapping(uint256 => uint256) public value;

		function test() public returns (uint256) {
			value[0] = 1;
			return value[0];
		}
	}
	"""
	code = SolcCompiler().compile(code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program(optimize=False)
	code = output_block.compile()
	assert isinstance(code, bytes)

def test_unchcked_math():
	# From https://solidity-by-example.org/unchecked-math/
	code = """
		contract UncheckedMath {
			function add(uint256 x, uint256 y) external pure returns (uint256) {
				// 22291 gas
				// return x + y;

				// 22103 gas
				unchecked {
					return x + y;
				}
			}

			function sub(uint256 x, uint256 y) external pure returns (uint256) {
				// 22329 gas
				// return x - y;

				// 22147 gas
				unchecked {
					return x - y;
				}
			}

			function sumOfCubes(uint256 x, uint256 y) external pure returns (uint256) {
				// Wrap complex math logic inside unchecked
				unchecked {
					uint256 x3 = x * x * x;
					uint256 y3 = y * y * y;

					return x3 + y3;
				}
			}
		}
	"""
	code = SolcCompiler().compile(code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program(optimize=False)
	code = output_block.compile()
	assert isinstance(code, bytes)

def test_should_handle_double_mapping():
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
	code = SolcCompiler().compile(code)
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program(optimize=False)
	code = output_block.compile()
	assert isinstance(code, bytes)

def test_coin_example():
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
	code = SolcCompiler().compile(code, CompilerSettings().optimize())
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program(optimize=False)
	output_block.create_plot()
	code = output_block.compile()
	assert isinstance(code, bytes)


def test_multicall():
	# https://github.com/EthereumClassicDAO/multicall3/blob/main/src/Multicall2.sol
	code = """
		contract Multicall2 {
			struct Call {
				address target;
				bytes callData;
			}

			struct Result {
				bool success;
				bytes returnData;
			}

			function aggregate(Call[] calldata calls) public returns (uint256 blockNumber, bytes[] memory returnData) {
				blockNumber = block.number;
				returnData = new bytes[](calls.length);
				for (uint256 i = 0; i < calls.length; i++) {
					(bool success, bytes memory ret) = calls[i].target.call(calls[i].callData);
					require(success, "Multicall aggregate: call failed");
					returnData[i] = ret;
				}
			}
		}
	"""
	code = SolcCompiler().compile(code, CompilerSettings().optimize())
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program(optimize=False)
	output_block.create_plot()
	code = output_block.compile()
	assert isinstance(code, bytes)

@pytest.mark.skip("Does not work atm.")
def test_wrapped_ether_example():
	# https://basescan.org/address/0x4200000000000000000000000000000000000006#code
	code = """
		contract WETH9 {
			string public name = "Wrapped Ether";
			string public symbol = "WETH";
			uint8 public decimals = 18;

			event Approval(address indexed src, address indexed guy, uint wad);
			event Transfer(address indexed src, address indexed dst, uint wad);
			event Deposit(address indexed dst, uint wad);
			event Withdrawal(address indexed src, uint wad);

			mapping(address => uint) public balanceOf;
			mapping(address => mapping(address => uint)) public allowance;

			fallback() external payable {
				deposit();
			}
			function deposit() public payable {
				balanceOf[msg.sender] += msg.value;
				emit Deposit(msg.sender, msg.value);
			}
			function withdraw(uint wad) public {
				require(balanceOf[msg.sender] >= wad);
				balanceOf[msg.sender] -= wad;
				payable(msg.sender).transfer(wad);
				emit Withdrawal(msg.sender, wad);
			}

			function totalSupply() public view returns (uint) {
				return address(this).balance;
			}

			function approve(address guy, uint wad) public returns (bool) {
				allowance[msg.sender][guy] = wad;
				emit Approval(msg.sender, guy, wad);
				return true;
			}

			function transfer(address dst, uint wad) public returns (bool) {
				return transferFrom(msg.sender, dst, wad);
			}

			function transferFrom(
				address src,
				address dst,
				uint wad
			) public returns (bool) {
				require(balanceOf[src] >= wad);

				if (src != msg.sender && allowance[src][msg.sender] != type(uint).max) {
					require(allowance[src][msg.sender] >= wad);
					allowance[src][msg.sender] -= wad;
				}

				balanceOf[src] -= wad;
				balanceOf[dst] += wad;

				emit Transfer(src, dst, wad);

				return true;
			}
		}
	"""
	code = SolcCompiler().compile(code, CompilerSettings().optimize())
	program = SsaProgramBuilder(
		execution=ProgramExecution.create_from_bytecode(code),
	)
	output_block = program.create_program(optimize=False)
	output_block.create_plot()
	code = output_block.compile()
	assert isinstance(code, bytes)

