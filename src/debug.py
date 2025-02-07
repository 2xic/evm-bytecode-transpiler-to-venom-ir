from test_utils.evm import run_vm
from test_utils.abi import encode_function_call


if __name__ == "__main__":
    value = "0x6080604052346100305736600411610030575f3560e01c636191b98214610034575f3560e01c63f8a8fd6d14610034575b5f5ffd5b604051439052604051806040516020010390f3"
    bytecode = bytes.fromhex(
        value.replace("0x","")
    )
    print((
        run_vm(bytecode, encode_function_call("fest()")).output.hex()
    ))
    print((
        run_vm(bytecode, encode_function_call("test()")).output.hex()
    ))
