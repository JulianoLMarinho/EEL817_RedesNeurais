import numpy as np
import funcoesAtivacao as fa
from RNA import *
from Camada import *

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
    entradaD = np.array(rna["entrada"])
    for camada in rna["camadas"]:
        entradaD = np.insert(entradaD, 0, 1)
        if 'peso' not in camada.keys():
            if ('neuronios' not in camada.keys()) or (camada['neuronios'] == None) or (not isinstance(camada['neuronios'], int)) or (camada['neuronios'] <= 0):
                return "Para camadas sem peso definido, o número de neurônios deve ser informado."
            camada['peso'] = getPesosAleatorios(camada['neuronios'], len(entradaD))
        matrizPeso = np.array(camada["peso"])       
        entradaD = np.dot(entradaD, matrizPeso.transpose())
        entradaD = fa.retornaFuncaoAtivacaoArray(entradaD, camada["funcaoAtivacao"])
    return entradaD

def getPesosAleatorios(linhas, colunas):
    return np.random.uniform(low=-1.0, high=1.0, size=(linhas, colunas))


camadas = []
camadas.append(Camada([[0.3, 0.1, 0.0], [-0.2, 0.3, 0.1], [-0.1, 0.0, 0.2]], {"nome": 'relu'}, 3))
camadas.append(Camada([[0.2, 0.3, -0.1, -0.1], [-0.3, 0.4, 0.1, 0.2]], {"nome": 'sigmoidal',"parametros": [5]}, 2))

rnaClass = RNA([0.8, 0.7], camadas)
print(camadas[0].peso)
print(rnaClass.camadas)

print(MLP(rna))
