#!/bin/bash

# Diretório com os arquivos de grafos
DIR="input"

# Caminho para o script Julia
JULIA_SCRIPT="linear_program.jl"

# Loop sobre todos os arquivos .txt na pasta
for file in "$DIR"/*.txt; do
    if [ -f "$file" ]; then
        echo "Processando $file..."
        # Chama o script Julia e imprime a saída no terminal
        julia "$JULIA_SCRIPT" "$file"
        echo "-----------------------------------"
    fi
done

