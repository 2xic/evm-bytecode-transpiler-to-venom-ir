from setuptools import setup, find_packages

setup(
	name="bytecode_transpiler",
	version="0.1",
	packages=find_packages(where="src"),
	package_dir={"": "src"},
	entry_points={
		"console_scripts": [
			"bytecode_transpiler=bytecode_transpiler.transpiler:main",
		],
	},
)
