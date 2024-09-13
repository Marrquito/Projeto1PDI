import numpy as np
import cv2

def ler_filtro(arquivo):
    with open(arquivo, 'r') as f:
        linhas = f.readlines()
    
    # Lendo as dimensões da matriz e o offset
    m, n, offset = map(int, linhas[0].split())
    
    # Lendo a matriz de filtro
    matriz = []
    for linha in linhas[1:]:
        matriz.append(list(map(int, linha.split())))
    
    matriz = np.array(matriz)
    return matriz, offset

def aplicar_correlacao_manual(canal, filtro, offset):
    # Dimensões do filtro
    altura_filtro, largura_filtro = filtro.shape
    # Dimensões do canal
    altura_canal, largura_canal = canal.shape
    
    # Calculando o padding necessário
    altura_padding = altura_filtro // 2
    largura_padding = largura_filtro // 2
    
    # Aplicando padding no canal
    canal_padded = np.pad(canal, ((altura_padding, altura_padding), (largura_padding, largura_padding)), mode='constant', constant_values=0)
    
    # Criando a imagem de saída
    resultado = np.zeros_like(canal, dtype=np.float32)
    
    # Aplicando a correlação manualmente
    for i in range(altura_canal):
        for j in range(largura_canal):
            # Região da imagem que será multiplicada pelo filtro
            region = canal_padded[i:i+altura_filtro, j:j+largura_filtro]
            # Multiplicação elemento a elemento e soma
            resultado[i, j] = np.sum(region * filtro) + offset
    
    # Aplicando o valor absoluto
    resultado = np.abs(resultado)
    
    # Normalizando os valores para a faixa [0, 1]
    resultado = resultado / np.max(resultado)
    
    # Expansão de histograma para a faixa [0, 255]
    resultado = resultado * 255
    resultado = np.clip(resultado, 0, 255)
    
    return resultado.astype(np.uint8)

def aplicar_correlacao(imagem, filtro, offset):
    # Separando os canais de cor
    canais = cv2.split(imagem)
    
    # Aplicando a correlação em cada canal
    canais_resultado = []
    for canal in canais:
        # Convertendo o canal para float32 para evitar overflow
        canal = canal.astype(np.float32)
        resultado = aplicar_correlacao_manual(canal, filtro, offset)
        canais_resultado.append(resultado)
    
    # Combinando os canais novamente
    imagem_resultado = cv2.merge(canais_resultado)
    return imagem_resultado

def aplicar_filtro_pontual(imagem):
    # Separando os canais de cor (BGR)
    canais = cv2.split(imagem)
    
    # Aplicando o filtro pontual em cada canal
    canais_resultado = []
    for canal in canais:
        # Obter as dimensões do canal
        altura, largura = canal.shape
        
        # Criar um novo array para armazenar o resultado
        canal_filtrado = np.zeros((altura, largura), dtype=np.uint8)
        
        # Iterar sobre cada pixel
        for i in range(altura):
            for j in range(largura):
                pixel = canal[i, j]
                if pixel <= 128:
                    valor_filtrado = 2 * pixel
                else:
                    valor_filtrado = 255 - 2 * (pixel - 128)
                
                # Garantir que o valor esteja no intervalo [0, 255]
                canal_filtrado[i, j] = np.clip(valor_filtrado, 0, 255)
        
        canais_resultado.append(canal_filtrado)
    
    # Combinando os canais novamente (BGR)
    imagem_resultado = cv2.merge(canais_resultado)
    return imagem_resultado

def main():
    # Caminhos para a imagem e o arquivo de filtro
    caminho_imagem = 'testpat1.jpg'
    caminho_filtro = 'filtro.txt'
    
    # Lendo a imagem
    imagem = cv2.imread(caminho_imagem)
    
    # Lendo o filtro e o offset
    filtro, offset = ler_filtro(caminho_filtro)
    
    # Aplicando a correlação
    imagem_correlacionada = aplicar_correlacao(imagem, filtro, offset)
    
    # Salvando a imagem resultante da correlação
    cv2.imwrite('imagem_correlacionada8.jpg', imagem_correlacionada)
    
    # Aplicando o filtro pontual
    imagem_filtro_pontual = aplicar_filtro_pontual(imagem)
    
    # Salvando a imagem resultante do filtro pontual
    cv2.imwrite('imagem_filtro_pontual13.jpg', imagem_filtro_pontual)

if __name__ == "__main__":
    main()
