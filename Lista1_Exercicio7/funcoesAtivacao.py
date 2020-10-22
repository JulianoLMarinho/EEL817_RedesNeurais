import math

def relu(x):
    if x<0: return 0
    return x

def tgh(x,args):
    return (math.exp(2*args[0]*x) - 1)/(math.exp(2*args[0]*x) + 1)

def linear(x, args):
    return args[0]*x + args[1]

def sigmoidal(x, args):
    return 1/(1 + math.exp(-args[0]*x))


def retornaFuncaoAtivacaoArray(lista, funcao):
    for i in range(0, len(lista)):
        lista[i] = retornaFuncaoAtivacao(lista[i], funcao)
    return lista

def retornaFuncaoAtivacao(valor, funcao):
    if funcao["nome"] == 'relu': return relu(valor)
    if funcao["nome"] == 'tgh': return tgh(valor, funcao["parametros"] or [1])
    if funcao["nome"] == 'linear': return linear(valor, funcao["parametros"] or [1,0])
    if funcao["nome"] == 'sigmoidal': return sigmoidal(valor, funcao["parametros"] or [1])
    else: return 0