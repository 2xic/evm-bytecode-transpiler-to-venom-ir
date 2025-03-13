## Setup
```bash
# git clone https://github.com/vyperlang/vyper
git clone https://github.com/harkal/vyper
git checkout f92e9a7e97643f49d6024db7a4904653612f8a7c

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

## Info
- https://en.wikipedia.org/wiki/Static_single-assignment_form
- https://x.com/harkal/status/1870054989990666584
- https://www.trailofbits.com/documents/RattleRecon.pdf
- https://github.com/vyperlang/vyper/blob/master/vyper/venom/README.md
- https://github.com/vyperlang/vyper/discussions/4513
