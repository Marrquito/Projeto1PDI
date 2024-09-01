from PIL import Image
import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self, caminho, caminho_saida):

        self.caminho        = caminho
        self.caminho_saida  = caminho_saida
        self.imagem         = self.abrir_imagem(caminho)
        
        self.m              = None
        self.n              = None
        self.ox             = None
        self.oy             = None
        self.filtro         = []
        
        self.ler_filtro("filtro.txt")

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
        else:
            print("Nenhuma imagem para exibir.")

    def salvar_imagem(self):
        if self.imagem is not None:
            self.imagem.save(self.caminho_saida)
            print(f"Imagem salva em {self.caminho_saida}")
        else:
            print("Nenhuma imagem para salvar.")
    
    def ler_filtro(self, filtro_path):
        with open(filtro_path, 'r') as f:
            self.m, self.n = map(int, f.readline().strip().split())
            
            self.ox, self.oy = map(int, f.readline().strip().split())
 
            for _ in range(self.m):
                linha = list(map(float, f.readline().split()))
                self.filtro.append(linha)
        
        print("m:{}, n:{}, ox:{}, oy:{}, filtro:{}".format(self.m, self.n, self.ox, self.oy, self.filtro))


if __name__ == "__main__":
    caminho_entrada = str(input("Digite o caminho da imagem: "))
    caminho_saida   = str(input("Digite o caminho de sáida: "))
    
    processador = ImageProcessor(caminho_entrada, caminho_saida)

    processador.exibir_imagem()

    processador.salvar_imagem()
