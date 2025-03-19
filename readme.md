# bytecode venom transpiler

Input is EVM bytecode and output is Venom IR.

## Setup
```bash
git clone https://github.com/vyperlang/vyper
git checkout 579dd5714145b15c772c8eb4066ade34a94ddef1
pip3 install -e .

python3 -m vyper.cli.venom_main
```

## Example
*todo*

## Looking at the ir
```bash
# Generates the Venom IR output
./generator generate [vyper file]
# Compiles the Venom IR output
./generator compile [venom file]
```

## Info
- https://en.wikipedia.org/wiki/Static_single-assignment_form
- https://x.com/harkal/status/1870054989990666584
- https://www.trailofbits.com/documents/RattleRecon.pdf
- https://github.com/vyperlang/vyper/blob/master/vyper/venom/README.md
- https://github.com/vyperlang/vyper/discussions/4513
