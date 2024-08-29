from PIL import Image
import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self, caminho, caminho_saida):

        self.caminho        = caminho
        self.caminho_saida  = caminho_saida
        self.imagem         = self.abrir_imagem(caminho)

    def abrir_imagem(self, caminho):
        try:
            imagem = Image.open(caminho)
            print(f"Imagem {caminho} aberta com sucesso!")
            
            return imagem
        except IOError:
            print("Erro ao abrir a imagem.")
            
            return None

    def exibir_imagem(self, imagem=None):
        if imagem is None:
            imagem = self.imagem

        if imagem is not None:
            plt.imshow(imagem)
            plt.axis("off")
            plt.show()
        else:
            print("Nenhuma imagem para exibir.")

    def salvar_imagem(self):
        if self.imagem is not None:
            self.imagem.save(self.caminho_saida)
            print(f"Imagem salva em {self.caminho_saida}")
        else:
            print("Nenhuma imagem para salvar.")


if __name__ == "__main__":
    caminho_entrada = str(input("Digite o caminho da imagem: "))
    caminho_saida   = str(input("Digite o caminho de sáida: "))
    
    processador = ImageProcessor(caminho_entrada, caminho_saida)

    processador.exibir_imagem()

    processador.salvar_imagem()
