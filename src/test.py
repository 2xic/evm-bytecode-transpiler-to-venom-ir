from test_utils.compiler import SolcCompiler
from test_utils.evm import execute_function
from test_utils.abi import encode_function_call
from transpiler import assert_compilation
from symbolic import EVM


def execute(bytecode_a, bytecode_b, function):
    out_a = execute_function(bytecode_a, function)
    out_b = execute_function(bytecode_b, function)
   # #print((out_a, out_b))
    assert out_a == \
            out_b, f"{out_a} != {out_b} with {function.hex()}"
    return True

def test_simple_hello_world():
    for via_ir in [True, False]:
        code = """
        contract Hello {
            function test() public returns (uint256) {
                return 1;
            }
        }
        """
        output = SolcCompiler().compile(code)
        transpiled = assert_compilation(output)
        
        assert execute(
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
    transpiled = assert_compilation(output)
    #print(transpiled.hex())
    
    assert execute(
        output,
        transpiled,
        encode_function_call("bagel()"),        
    )
    # TODO: Figure out what the issue is.
    assert execute(
        output,
        transpiled,
        encode_function_call("test()"),        
    )

def test_nested_if_conditions():
    code = """
    contract Hello {
        function test() public returns (uint256) {
            if (block.number > 10) {        
                return 2;
            }
        }
    }
    """
    output = SolcCompiler().compile(code)
    transpiled = assert_compilation(output)
    #print(transpiled.hex())
    
    assert execute(
        output,
        transpiled,
        encode_function_call("test()"),        
    )

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
    output = SolcCompiler().compile(code)
    transpiled = assert_compilation(output)
    
    assert execute(
        output,
        transpiled,
        encode_function_call("test()"),        
    )

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
    output = SolcCompiler().compile(code)
    transpiled = assert_compilation(output)
    
    assert execute(
        output,
        transpiled,
        encode_function_call("test()"),        
    )

    code = """
    contract Hello {
        function test() public returns (uint256) {
            return a(15);
        }

        function fest() public returns (uint256) {
            return a(5);
        }

        function a(uint256 a) internal returns (uint256){
            if (a > 10) {        
                return 2;
            }
        }
    }
    """
    output = SolcCompiler().compile(code)
    transpiled = assert_compilation(output)
    
    assert execute(
        output,
        transpiled,
        encode_function_call("test()"),        
    )


def test_conditions():
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
    output = SolcCompiler().compile(code)
    transpiled = assert_compilation(output)
    
    assert execute(
        output,
        transpiled,
        encode_function_call("test()"),        
    )
    assert len(transpiled) < len(output)

def test_multiple_functions():
    code = """
    contract Hello {
        function test() public returns (uint256) {
            return block.timestamp;
        }

        function fest() public returns (uint256) {
            return block.number;
        }
    }
    """
    output = SolcCompiler().compile(code)
    transpiled = assert_compilation(output)
    
    assert execute(
        output,
        transpiled,
        encode_function_call("test()"),        
    )
    assert len(transpiled) < len(output)
    assert execute(
        output,
        transpiled,
        encode_function_call("fest()"),        
    )
    assert len(transpiled) < len(output)

def test_counter_contract_example():
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
    output = SolcCompiler().compile(code)
    transpiled = assert_compilation(output)
    assert len(transpiled) < len(output)
    
    assert execute(
        output,
        transpiled,
        encode_function_call("incrementCounter()"),        
    )
    assert execute(
        output,
        transpiled,
        encode_function_call("decrementCounter()"),        
    )
    assert execute(
        output,
        transpiled,
        encode_function_call("getCount()"),        
    )

# TODO: need to support sstore opcodes?
def skip_test_simple_mapping_example():
    code = """
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
    output = SolcCompiler().compile(code)
    transpiled = assert_compilation(output)
    assert len(transpiled) < len(output)
    
    assert execute(
        output,
        transpiled,
        encode_function_call("setResults(address)", ["address"], ['0x' + 'ff' * 20]),
    )
    #assert execute(
    #    output,
    #    transpiled,
    #    encode_function_call("nonExistingMethod()"),        
    #)
    #assert execute(
    #    #output,
    #    #transpiled,
    #    #encode_function_call("getResults(address)", ["address"], ['0x' + 'ff' * 20]),
    #)

def skip_test_example_coin_contract():
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
    output = SolcCompiler().compile(code)
    transpiled = assert_compilation(output)
    assert len(transpiled) < len(output)
    
    assert execute(
        output,
        transpiled,
        encode_function_call("nonExistingMethod()"),
    )
    assert execute(
        output,
        transpiled,
        encode_function_call("mint(address,uint256)", ["address", "uint"], ['0x' + 'ff' * 20, 10000]),
    )
    assert execute(
        output,
        transpiled,
        encode_function_call("send(address,uint256)", ["address", "uint"], ['0x' + 'ff' * 20, 0]),
    )

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

