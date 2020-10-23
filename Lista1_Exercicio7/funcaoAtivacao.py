import funcoesAtivacao as fa
import math

class FuncaoAtivacao:

    def __init__(self, nome, parametros = []):
        self.nome = nome
        self.parametros = parametros
    
    def relu(self, x):
        if x<0: return 0
        return x

    def tgh(self, x,args):
        return (math.exp(2*args[0]*x) - 1)/(math.exp(2*args[0]*x) + 1)

    def linear(self, x, args):
        return args[0]*x + args[1]

    def sigmoidal(self, x, args):
        return 1/(1 + math.exp(-args[0]*x))


    def retornaFuncaoAtivacaoArray(self, lista):
        for i in range(0, len(lista)):
            lista[i] = self.retornaFuncaoAtivacao(lista[i])
        return lista

    def retornaFuncaoAtivacao(self, valor):
        if self.nome == 'relu': return self.relu(valor)
        if self.nome == 'tgh': return self.tgh(valor, self.parametros or [1])
        if self.nome == 'linear': return self.linear(valor, self.parametros or [1,0])
        if self.nome == 'sigmoidal': return self.sigmoidal(valor, self.parametros or [1])
        else: return 0
