from vyper.compiler.phases import generate_bytecode
from vyper.compiler.settings import OptimizationLevel
from vyper.venom import generate_assembly_experimental, run_passes_on
from vyper.venom.check_venom import check_venom_ctx
from vyper.venom.parser import parse_venom

GAS_OPTIMIZATION_LEVEL = OptimizationLevel.GAS
CODE_OPTIMIZATION_LEVEL = OptimizationLevel.CODESIZE
DEFAULT_OPTIMIZATION_LEVEL = OptimizationLevel.default()
NO_OPTIMIZATION = OptimizationLevel.NONE


def compile_venom(
	venom_source, optimization_level: OptimizationLevel = OptimizationLevel.default()
):
	ctx = parse_venom(venom_source)
	check_venom_ctx(ctx)
	run_passes_on(ctx, optimization_level)
	asm = generate_assembly_experimental(ctx)
	bytecode = generate_bytecode(asm, compiler_metadata=None)
	return bytecode
