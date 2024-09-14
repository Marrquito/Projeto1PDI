import numpy as np
import cv2


class ImageProcessor:
    def __init__(self, caminho_img, caminho_filtro):

        self.caminho_img    = caminho_img
        self.caminho_filtro = caminho_filtro
        
        self.imagem         = self.abrir_imagem()
        
        if self.abrir_imagem == None:
            return 
        
        self.m              = None
        self.n              = None
        self.offset         = None
        self.filtro         = []
        
        self.ler_filtro("filtro.txt")

    def abrir_imagem(self):
        try:
            imagem = cv2.imread(self.caminho_img)
            
            return imagem
        except IOError:
            print("Erro ao abrir a imagem.")
            
            return None

    def salvar_imagem(self, img, caminho_saida):
        if img is not None:
            cv2.imwrite(caminho_saida, img)
        else:
            print("Nenhuma imagem para salvar.")
    
    def ler_filtro(self, filtro_path):
        with open(filtro_path, 'r') as f:
            self.m, self.n, self.offset = map(int, f.readline().strip().split())
 
            for _ in range(self.m):
                linha = list(map(float, f.readline().split()))
                self.filtro.append(linha)
        
        self.filtro = np.array(self.filtro)
    
    def aplicar_correlacao_manual(self, canal):
        # Dimensões do filtro
        altura_filtro, largura_filtro = self.filtro.shape
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
                vizinhanca = canal_padded[i:i+altura_filtro, j:j+largura_filtro]
                # Multiplicação elemento a elemento e soma
                resultado[i, j] = np.sum(vizinhanca * self.filtro) + self.offset
        
        # Aplicando o valor absoluto
        resultado = np.abs(resultado)
        
        # Normalizando os valores para a faixa [0, 1]
        resultado = resultado / np.max(resultado)
        
        # Expansão de histograma para a faixa [0, 255]
        resultado = resultado * 255
        resultado = np.clip(resultado, 0, 255)
        
        return resultado.astype(np.uint8)
    
    def aplicar_correlacao(self):
        # Aplicando a correlação em cada canal
        canais = cv2.split(self.imagem)
        canais_resultado = []
        
        for canal in canais:
            # Convertendo o canal para float32 para evitar overflow
            canal = canal.astype(np.float32)
            resultado = self.aplicar_correlacao_manual(canal)
            canais_resultado.append(resultado)
        
        # Combinando os canais novamente
        imagem_resultado = cv2.merge(canais_resultado)
        return imagem_resultado

    def aplicar_filtro_pontual(self):
        # Separando os canais de cor (BGR)
        canais = cv2.split(self.imagem)
        
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
    
    def rgb_to_yiq(self):
        # Matriz de transformação de RGB para YIQ
        rgb_to_yiq_matrix = np.array([[0.299, 0.587, 0.114],
                                    [0.596, -0.274, -0.322],
                                    [0.211, -0.523, 0.312]])

        # Converta a imagem RGB para float32 para evitar problemas de precisão
        imagem_rgb = self.imagem.astype(np.float32) / 255.0

        # Aplique a matriz de transformação
        imagem_yiq = np.dot(imagem_rgb, rgb_to_yiq_matrix.T)

        return imagem_yiq

    def yiq_to_rgb(self, imagem_yiq):
        # Matriz de transformação de YIQ para RGB
        yiq_to_rgb_matrix = np.array([[1.0, 0.956, 0.621],
                                    [1.0, -0.272, -0.647],
                                    [1.0, -1.106, 1.703]])

        # Aplica a matriz de transformação
        imagem_rgb = np.dot(imagem_yiq, yiq_to_rgb_matrix.T)

        # Garante que os valores estejam no intervalo [0, 1]
        imagem_rgb = np.clip(imagem_rgb, 0, 1)

        # Converte para uint8
        imagem_rgb = (imagem_rgb * 255).astype(np.uint8)

        return imagem_rgb

    # Substitui a banda Y original pela filtrada
    def substituir_banda_y(self, imagem_yiq, banda_y_filtrada):
        imagem_yiq[:, :, 0] = banda_y_filtrada
        return imagem_yiq

    def get_y_band(self, imagem_yiq):
        # A banda Y é o primeiro canal da imagem YIQ
        banda_y = imagem_yiq[:, :, 0]
        return banda_y
    
    def aplicar_filtro_pontual_banda_y(self):
        # Convertendo a imagem para o espaço de cor YIQ
        imagem_yiq = self.rgb_to_yiq()
        
        # Extrai a banda Y
        banda_y = self.get_y_band(imagem_yiq)
       
        # Cria uma matriz vazia com o mesmo tamanho da banda Y para armazenar o resultado
        banda_y_filtrada = np.zeros_like(banda_y, dtype=np.float32)
        
        # Aplica a primeira parte do filtro: y = 2x para valores de 0 a 128
        banda_y_filtrada[banda_y <= 128] = 2 * banda_y[banda_y <= 128]
        
        # Aplica a segunda parte do filtro: y = 255 - 2*(128 - x) para valores acima de 128
        banda_y_filtrada[banda_y > 128] = 255 - 2 * (128 - banda_y[banda_y > 128])
        
        # Garante que os valores filtrados estejam na faixa [0, 255]
        banda_y_filtrada = np.clip(banda_y_filtrada, 0, 255)
        
        # Substitui a banda Y na imagem YIQ pela banda filtrada
        imagem_yiq_com_filtrada = self.substituir_banda_y(imagem_yiq, banda_y_filtrada)
        
        # Converte a imagem YIQ de volta para RGB
        imagem_rgb_final = self.yiq_to_rgb(imagem_yiq_com_filtrada)
        
        return imagem_rgb_final
    
if __name__ == "__main__":
    caminho_entrada = str(input("Digite o caminho da imagem: "))
    caminho_filtro  = str(input("Digite o caminho do filtro: "))
    
    processador = ImageProcessor(caminho_entrada, caminho_filtro)
    
    imagem_correlacionada = processador.aplicar_correlacao()
    processador.salvar_imagem(imagem_correlacionada, "resultado_img_correlacionada.jpg")
    
    imagem_filtro_pontual = processador.aplicar_filtro_pontual()
    processador.salvar_imagem(imagem_filtro_pontual, "resultado_img_filtro_pontual.jpg")
    
    imagem_filtro_pontual_banda_y = processador.aplicar_filtro_pontual_banda_y()
    processador.salvar_imagem(imagem_filtro_pontual_banda_y, "resultado_img_filtro_pontual_banda_y.jpg")
    
