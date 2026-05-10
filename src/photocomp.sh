#!/bin/bash

# Muda aqui para "cuda" ou "openmp" consoante o que queres testar
VERSAO="openmp"

if [ "$VERSAO" == "openmp" ]; then
    EXEC="./sgmOpenMP"
else
    EXEC="./sgmCuda"
fi

# Lista com os nomes base das imagens
IMAGENS=("bull" "cones" "teddy" "venus")

echo -e "\n===================================================================="
echo "A iniciar os testes..."
echo "===================================================================="

for img in "${IMAGENS[@]}"; do
    # 1. Nomes dos ficheiros de Input
    img_esq="l${img}.pgm"
    img_dir="r${img}.pgm"
    
    # 2. Nomes dos ficheiros de Output Esperados
    out_host="h_d${img}_${VERSAO}.pgm"
    out_device="d_d${img}_${VERSAO}.pgm"

    echo -e "\n---------------------------------------------------------------------------"
    echo " [1] A executar $EXEC para a imagem '$img'..."
    echo "---------------------------------------------------------------------------"   
   
    # 3. Correr o executável forçando os nomes de output corretos
    $EXEC -l "$img_esq" -r "$img_dir" -t "$out_host" -o "$out_device"
    
    echo -e "\n---------------------------------------------------------------------------"
    echo " [2] A comparar diferenças..."
    echo "---------------------------------------------------------------------------"
    
    # 4. Executar o testDiffs
    if [ -f "$out_host" ] && [ -f "$out_device" ]; then
        ./testDiffs "$out_host" "$out_device"
    else
        echo "ERRO: O ficheiro $out_host ou $out_device não foi encontrado."
        echo "O programa falhou a processar a imagem $img."
    fi
    
    echo -e "\n======================================================================"
done

echo "Testes concluídos!"
