## Setup
```bash
git clone https://github.com/vyperlang/vyper
git checkout 537313b0dd47b3c086fa46d1ef7d8282101fa128

pip3 install -e .

python3 -m vyper.cli.venom_main
```

## Looking at the ir
```bash
# Generates the Venom IR output
./generator generate [vyper file]
# Compiles the Venom IR output
./generator compile [venom file]
```

## TODO
- need to fix how to handle phi functions where the value differs based on parent block

## Info
- https://en.wikipedia.org/wiki/Static_single-assignment_form
- https://x.com/harkal/status/1870054989990666584
- https://www.trailofbits.com/documents/RattleRecon.pdf
- 