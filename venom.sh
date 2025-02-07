#!/bin/bash

usage() {
    echo "Usage: $0 {generate|compile} <file>"
    exit 1
}

if [[ $# -ne 2 ]]; then
    usage
fi

command=$1
file=$2

if [[ "$command" == "generate" ]]; then
    echo "Generating from file: $file"
    vyper --experimental-codegen -f bb_runtime $file > output.venom
elif [[ "$command" == "compile" ]]; then
    echo "Compiling file: $file"
    python3 -m vyper.cli.venom_main $file
else
    echo "Invalid command: $command"
    usage
fi
