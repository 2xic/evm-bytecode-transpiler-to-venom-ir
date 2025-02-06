from test_utils.compiler import SolcCompiler
from test_utils.evm import run_vm
from test_utils.abi import encode_function_call
from transpiler import assert_compilation


def execute(bytecode_a, bytecode_b, function):
    out_a = run_vm(bytecode_a, function).output.hex()
    out_b = run_vm(bytecode_b, function).output.hex()
   # print((out_a, out_b))
    assert out_a == \
            out_b, f"{out_a} != {out_b} with {function.hex()}"
    return True

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
    assert len(transpiled) < len(output)
#    raise Exception(f"{len(transpiled) } < {len(output)}")

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
    print(transpiled.hex())
    
    assert execute(
        output,
        transpiled,
        encode_function_call("bagel()"),        
    )
    # TODO: Figure out what the issue is.
#    assert execute(
#        output,
#        transpiled,
#        encode_function_call("test()"),        
#    )

def skip_test_nested_if_conditions():
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
    print(transpiled.hex())
    
    assert execute(
        output,
        transpiled,
        encode_function_call("test()"),        
    )

