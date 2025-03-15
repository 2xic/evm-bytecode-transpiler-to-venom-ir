from cfg import create_cfg
from test_utils.compiler import SolcCompiler

def test():
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
	compiled = SolcCompiler().compile(code, via_ir=False)
	create_cfg(compiled)
