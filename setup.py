from setuptools import setup, find_packages

install_requires = [
    "ordered_set==4.1.0",
    "py-solc-x==2.0.3",
    "graphviz==0.20.1",
    "vyper @ git+https://github.com/vyperlang/vyper.git@642da76356bf85b599e7c455d5b1e2dd3722a6f4",
    "py-evm==0.10.1b2",
    "pysha3==1.0.2",
    "eth-abi==4.2.1",
]

dev_requires = [
    "matplotlib==3.10.1",
    "pytest==8.3.5",
]

setup(
    name="bytecode_transpiler",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
    },
    entry_points={
        "console_scripts": [
            "bytecode_transpiler=bytecode_transpiler.transpiler:main",
        ],
    },
)