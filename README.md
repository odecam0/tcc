# Personalização em Reconhecimento de Atividades Humanas

<div align="center">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="80px"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original-wordmark.svg" height="80px"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original-wordmark.svg" height="80px"/>
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original-wordmark.svg" height="80px"/>
<img src="/readme-assets/Scikit_learn_logo_small.svg" height="40px"/>
</div>

<br />
                   
Este repositório contém o código do projeto desenvolvido durante o meu Trabalho de Conclusão de Curso (TCC), onde explorei diferentes abordagens de reconhecimento de atividades humanas utilizando Python e métodos de aprendizado de máquina em séries temporais.

O código surgiu conforme os scripts de experimentos realizados foram ficando cada vez mais complexos de manter, então, após diversas re-estruturações, chegou nesta estrutura.

## Organização do código

O código se encontra dentro da pasta [/src](/src/).
O arquivo [ARPC.py](/src/ARPC.py) contém o esqueleto principal,
e define uma classe Arpc que implementa métodos para cada etapa do processo
de reconhecimento de atividades: 

<img src="/images/arp_sequence.png" />

- Pré-processamento
- Janelamento de dados
- Extração de características
- Classificação
Também foram adcionados métodos para realizar aumento de dados.

A classe possui estado para cada passo neste processo.
Possui variáveis que armazenam:
- Dados brutos como um DataFrame do pandas.
- Dados pré-processados como um DataFrame do pandas.
- Janelas de dados <TODO: Relembrar como estes dados estão estruturados>
- Características extraídas dos dados segmentados.
- Modelos de aprendizado de máquinas treinados
- Matrizes de confusão com resultados de classificações.

Também possui uma variável com uma referência a um possível experimento anterior, que é apenas mais um objeto desta mesma classe.

Funções que permitem comparar diferentes experimentos estruturados desta forma foram implementadas em [arpc_plot.py](/src/arpc_plot.py).

## Pontos interessantes

- Este projeto foi construido visando ser flexível.
  Em diversos pontos o comportamento do sistema é determinado pelos parâmetros passados para as funções,
  por este motivo existem muitas funções de segunda ordem.

- Foi implementado uma maneira simples de realizar aumento de dados em janelas de dados.
  Ao passar uma lista de funções para Arpc.apply_each_window, o objeto retornado por estas funções
  será adcionado ao conjunto de janelas já armazenadas no objeto Arpc, preservando o rótulo associado.
  
  Uma função que implementa um aumento de dados foi feita em [TimeWarpWindow.py](/src/TimeWarpWindow.py). 
  Ela deforma a série temporal esticando e comprimindo determinadas partes, simulando o fato de que duas
  realizações de uma mesma atividade podem ser feitas com períodos diferentes de duração de seus movimentos.
  Este método é exemplificado na seguinte imagem:
  
  <img src="/images/time_warping.png" />
  
  Um exemplo da utilização deste método se encontra nestes dois arquivos: [exp_time_w.py](/src/experiment_time_warping/exp_time_w.py), [plotting_given_arpos.py](/src/experiment_time_warping/plotting_given_arpos.py)
  
  Que geram o seguinte gráfico:
  
  <img src="/src/experiment_time_warping/1st_plot_commented.png" />

O corpo deste código possui 1418 linhas no total.
