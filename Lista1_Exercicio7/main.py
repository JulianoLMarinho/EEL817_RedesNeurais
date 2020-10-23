import numpy as np
from RNA import *
from Camada import *
from funcaoAtivacao import *

#arrayPesos = [[[-0.3, -0.1, 0.0],[0.2, 0.1, 0.4]], [[0.3, 0.0, -0.1],[0.4,0.0,0.4],[0.1,0.3,0.0]], [[0.3,-0.1,0.5,-0.1],[0.4,0.3,-0.2,0.4]]]
arrayPesos = [[[0.3, 0.1, 0.0], [-0.2, 0.3, 0.1], [-0.1, 0.0, 0.2], [-0.1, 0.0, 0.2]], [[0.2, 0.3, -0.1, -0.1], [-0.3, 0.4, 0.1, 0.2]]]
entrada = np.array([-0.3, 0.5])

'''
rna = {
    "entrada": [-0.3, 0.5],
    "camadas":[
        {
            "peso": [[-0.3, -0.1, 0.0],[0.2, 0.1, 0.4]],
            "funcaoAtivacao": {
                "nome": 'sigmoidal',
                "parametros": [7]
            },
            "neuronios": 2
        },
        {
            "peso": [[0.3, 0.0, -0.1],[0.4,0.0,0.4],[0.1,0.3,0.0]],
            "funcaoAtivacao": {
                "nome": 'tgh',
                "parametros": [1]
            },
            "neuronios": 3
        },
        {
            "peso": [[0.3,-0.1,0.5,-0.1],[0.4,0.3,-0.2,0.4]],
            "funcaoAtivacao": {
                "nome": 'linear',
                "parametros": [1,0]
            },
            "neuronios": 2
        }
    ]
}
'''
rna = {
    "entrada": [0.8, 0.7],
    "camadas":[
        {
            "peso": [[0.3, 0.1, 0.0], [-0.2, 0.3, 0.1], [-0.1, 0.0, 0.2]],
            "funcaoAtivacao": {
                "nome": 'relu'
            },
            "neuronios": 3
        },
        {
            "peso": [[0.2, 0.3, -0.1, -0.1], [-0.3, 0.4, 0.1, 0.2]],
            "funcaoAtivacao": {
                "nome": 'sigmoidal',
                "parametros": [5]
            },
            "neuronios": 2
        }
    ]
}

def MLP(rna):
    entradaD = np.array(rna.entrada)
    for camada in rna.camadas:
        entradaD = np.insert(entradaD, 0, 1)
        #print(entradaD)
        if camada.peso == None:
            if (camada.neuronios == None) or (not isinstance(camada.neuronios, int)) or (camada.neuronios <= 0):
                return "Para camadas sem peso definido, o número de neurônios deve ser informado."
            camada.peso = getPesosAleatorios(camada.neuronios, len(entradaD))
        matrizPeso = np.array(camada.peso)  
        entradaD = np.dot(entradaD, matrizPeso.transpose())
        entradaD = camada.funcaoAtivacao.retornaFuncaoAtivacaoArray(entradaD)
    return entradaD

def getPesosAleatorios(linhas, colunas):
    return np.random.uniform(low=-1.0, high=1.0, size=(linhas, colunas))


camadas = []
camadas.append(Camada(FuncaoAtivacao('relu'), [[0.3, 0.1, 0.0], [-0.2, 0.3, 0.1], [-0.1, 0.0, 0.2]]))
camadas.append(Camada(FuncaoAtivacao('sigmoidal',[5]), [[0.2, 0.3, -0.1, -0.1], [-0.3, 0.4, 0.1, 0.2]]))

#camadas.append(Camada(FuncaoAtivacao('relu'), 3))
#camadas.append(Camada(FuncaoAtivacao('sigmoidal',[5]), 2))

rnaClass = RNA([0.8, 0.7], camadas)
print(MLP(rnaClass))
