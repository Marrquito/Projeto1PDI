import numpy as np
import cv2
from PIL import Image

def ler_filtro(arquivo):
    with open(arquivo, 'r') as f:
        linhas = f.readlines()
    
    # Lendo as dimensões da matriz e o offset
    m, n, offset = map(int, linhas[0].split())
    
    # Lendo a matriz de filtro
    matriz = []
    for linha in linhas[1:]:
        matriz.append(list(map(float, linha.split())))
    
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
            vinzinhaca = canal_padded[i:i+altura_filtro, j:j+largura_filtro]
            # Multiplicação elemento a elemento e soma
            resultado[i, j] = np.sum(vinzinhaca * filtro) + offset
    
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

def rgb_para_yiq(imagem_rgb):
    # Converter a imagem RGB para YIQ
    imagem_yiq = np.zeros_like(imagem_rgb, dtype=np.float32)
    for x in range(imagem_rgb.shape[0]):
        for y in range(imagem_rgb.shape[1]):
            r, g, b = imagem_rgb[x, y] / 255.0
            Y = 0.299 * r + 0.587 * g + 0.114 * b
            I = 0.596 * r - 0.274 * g - 0.322 * b
            Q = 0.211 * r - 0.523 * g + 0.312 * b
            imagem_yiq[x, y] = [Y, I, Q]
    return imagem_yiq

def yiq_para_rgb(imagem_yiq):
    # Converter a imagem YIQ de volta para RGB
    imagem_rgb = np.zeros_like(imagem_yiq, dtype=np.float32)
    for x in range(imagem_yiq.shape[0]):
        for y in range(imagem_yiq.shape[1]):
            Y, I, Q = imagem_yiq[x, y]
            r = Y + 0.956 * I + 0.621 * Q
            g = Y - 0.272 * I - 0.647 * Q
            b = Y - 1.106 * I + 1.703 * Q
            imagem_rgb[x, y] = [r, g, b]
    return np.clip(imagem_rgb * 255, 0, 255).astype(np.uint8)

def aplicar_filtro_pontual_na_banda_y(imagem_rgb):
    # Converter a imagem RGB para YIQ
    imagem_yiq = rgb_para_yiq(imagem_rgb)
    
    # Separar as bandas Y, I e Q
    Y, I, Q = imagem_yiq[:, :, 0], imagem_yiq[:, :, 1], imagem_yiq[:, :, 2]
    
    # Aplicar o filtro pontual na banda Y
    Y_filtrada = np.zeros_like(Y, dtype=np.float32)
    altura, largura = Y.shape
    for x in range(altura):
        for y in range(largura):
            pixel = Y[x, y]
            if pixel <= 0.5:
                valor_filtrado = 2 * pixel
            else:
                valor_filtrado = 1 - 2 * (pixel - 0.5)
            Y_filtrada[x, y] = np.clip(valor_filtrado, 0, 1)
    
    # Criar uma lista com Y_filtrada, I e Q
    bandas = [Y_filtrada, I, Q]
    
    # Combinar as bandas Y filtrada, I e Q novamente
    imagem_yiq_filtrada = np.stack(bandas, axis=-1)
    
    # Converter a imagem YIQ de volta para RGB
    imagem_rgb_filtrada = yiq_para_rgb(imagem_yiq_filtrada)
    
    return imagem_rgb_filtrada

def main():
    # Caminhos para a imagem e o arquivo de filtro
    caminho_imagem = 'testpat1k.tif'
    caminho_filtro = 'filtro.txt'
    
    # Lendo a imagem
    imagem = cv2.imread(caminho_imagem)
    #imagem_pil = Image.open('Shapes.png')
    #imagem_rgb = np.array(imagem_pil)
    
    # Lendo o filtro e o offset
    filtro, offset = ler_filtro(caminho_filtro)
    
    # Aplicando a correlação
    imagem_correlacionada = aplicar_correlacao(imagem, filtro, offset)

    
    # Salvando a imagem resultante da correlação
    cv2.imwrite('imagem_correlacionada_p4.tif', imagem_correlacionada)
    
    # Aplicando o filtro pontual
    imagem_filtro_pontual = aplicar_filtro_pontual(imagem)
    
    # Salvando a imagem resultante do filtro pontual
    cv2.imwrite('imagem_filtro_pontual_p4.tif', imagem_filtro_pontual)

    #imagem_Y = aplicar_filtro_pontual_na_banda_y(imagem_rgb)
    #imagem_Y_pill = Image.fromarray(imagem_Y)
    #imagem_Y_pill.save('imagem_filtro_Y_1.png')

if __name__ == "__main__":
    main()
