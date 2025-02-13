## Setup
```bash
git clone https://github.com/vyperlang/vyper
git checkout c75a2da09aeaa49444c4a9d3489b0557a829862b

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
- https://github.com/vyperlang/vyper/blob/master/vyper/venom/README.md
