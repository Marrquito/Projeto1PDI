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
    f_h, f_w = filtro.shape
    # Dimensões do canal
    c_h, c_w = canal.shape
    
    # Calculando o padding necessário
    pad_h = f_h // 2
    pad_w = f_w // 2
    
    # Aplicando padding no canal
    canal_padded = np.pad(canal, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Criando a imagem de saída
    resultado = np.zeros_like(canal, dtype=np.float32)
    
    # Aplicando a correlação manualmente
    for i in range(c_h):
        for j in range(c_w):
            # Região da imagem que será multiplicada pelo filtro
            region = canal_padded[i:i+f_h, j:j+f_w]
            # Multiplicação elemento a elemento e soma
            resultado[i, j] = np.sum(region * filtro) + offset
    
    # Normalizando os valores para a faixa [0, 255]
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
    # Separando os canais de cor
    canais = cv2.split(imagem)
    
    # Aplicando o filtro pontual em cada canal
    canais_resultado = []
    for canal in canais:
        # Aplicando a expressão y=2x para valores de 0 a 128
        canal = np.where(canal <= 128, 2 * canal, canal)
        # Aplicando a expressão y=255+2*(128-x) para valores acima de 128
        canal = np.where(canal > 128, 255 + 2 * (128 - canal), canal)
        canais_resultado.append(canal)
    
    # Combinando os canais novamente
    imagem_resultado = cv2.merge(canais_resultado)
    return imagem_resultado

def main():
    # Caminhos para a imagem e o arquivo de filtro
    caminho_imagem = 'ryuk.jpg'
    caminho_filtro = 'filtro.txt'
    
    # Lendo a imagem
    imagem = cv2.imread(caminho_imagem)
    
    # Lendo o filtro e o offset
    filtro, offset = ler_filtro(caminho_filtro)
    
    # Aplicando a correlação
    imagem_correlacionada = aplicar_correlacao(imagem, filtro, offset)
    
    # Salvando a imagem resultante da correlação
    cv2.imwrite('imagem_correlacionada.jpg', imagem_correlacionada)
    
    # Aplicando o filtro pontual
    imagem_filtro_pontual = aplicar_filtro_pontual(imagem)
    
    # Salvando a imagem resultante do filtro pontual
    cv2.imwrite('imagem_filtro_pontual.jpg', imagem_filtro_pontual)

if __name__ == "__main__":
    main()
