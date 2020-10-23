class Camada:

    def __init__(self, funcaoAtivacao, neuroniosOuCamadas):
        self.funcaoAtivacao = funcaoAtivacao
        if isinstance(neuroniosOuCamadas, int):
            self.neuronios = neuroniosOuCamadas
            self.peso = None
        elif isinstance(neuroniosOuCamadas, list):
            self.peso = neuroniosOuCamadas 
            self.neuronios = None
    