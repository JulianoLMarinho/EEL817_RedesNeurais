# Lista 1 - Exercício 7

Para representar a RNA, foram criadas as classes RNA, Camada e FuncaoAtivacao.

* Classe FuncaoAtivacao
```python
class FuncaoAtivacao:
    def __init__(self, nome, parametros = []):
        self.nome = nome
        self.parametros = parametros
```
* Classe Camada
```python
class Camada:

    def __init__(self, funcaoAtivacao, neuroniosOuCamadas):
        self.funcaoAtivacao = funcaoAtivacao
        if isinstance(neuroniosOuCamadas, int):
            self.neuronios = neuroniosOuCamadas
            self.peso = None
        elif isinstance(neuroniosOuCamadas, list):
            self.peso = neuroniosOuCamadas 
            self.neuronios = None
```
* Classe RNA
```python
class RNA:    
    def __init__(self, entrada, camadas):
        self.entrada = entrada
        self.camadas = camadas
```

Para a classe Camada, no momento da instanciação usuário pode optar por informar os pesos da camada, ou informar o número de neurônios na camada. Quando é informado apenas o número de neurônios, uma matriz de pesos aleatória é gerada seguinto uma distribuição uniforme com intervalos de -1 a 1.

A FuncaoAtivacao receberá o nome da função e os parâmetros (quando houver). Caso a função informada deva receber parâmetros, porém os parâmetros não foram informados, as operações irão utilizar valores default para cada nome de função: a=1 e b=0.





