from test_utils.compiler import SolcCompiler
from test_utils.evm import run_vm
from test_utils.abi import encode_function_call
from transpiler import assert_compilation


def execute(bytecode_a, bytecode_b, function):
    return run_vm(bytecode_a, function).output.hex() == \
            run_vm(bytecode_b, function).output.hex()

def test_simple_hello_world():
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

def skip_test_simple_multiple_functions():
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
    
    assert execute(
        output,
        transpiled,
        encode_function_call("test()"),        
    )

