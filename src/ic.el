----------------------------------------------------------
----------- Sobre iniciação científica -------------------

https://mail.google.com/mail/u/0/#search/alessandro_copetti%40id.uff.br

----------------------------------------------------------
----------- Conclusão ------------------------------------

Experimentos mostraram que a inserção de uma pequena parcela dos dados de um usuário, no conjunto de treinamento, contendo dados de outros usuários melhora consideravelmente a performance do modelo quando avaliados nos dados do usuário final, sendo um primeiro passo para personalização. Esta inserção de dados parece ser mais efetiva conforme a atividade envolva menos movimentos.

A intensidade onde houve mais dificuldade de generalizar foi a 'moderado', possívelmente pela definição do comportamento na Tabela 1 envolver algumas descisões por parte do participante na coleta de dados.

-----------------------------------------------------------
---------- Auto avaliação ---------------------------------


Adquiri bastante familiaridade com ferramentas utilizadas para manipulação e visualização de dados e também para especificação e treinamento de modelos de aprendizado de máquina. Acabei desenvolvedo mais facilidade em buscar oque preciso saber para utilizar alguma ferramenta e entender como esta funciona, buscando sua documentação e materiais relacionados.

Aprendi sobre como ocorre uma pesquisa científica, e sobre como pode ocorrer a colaboração entre colegas
pesquisando um mesmo tema.

Meu maior desafio foi organizar tudo que entendi e produzi durante o projeto para ser apresentado de forma textual, outro foi saber com clareza oque deveria ser feito em seguida em alguns momentos.

No final das contas agregou bastante ao meu conhecimento técnico no paradigma de programação 'Data Driven', e me introduziu à escrita científica.

-----------------------------------------------------------

<<Mostrar a matriz de confusão para todos os 11 participantes
-- para alguns participantes os dados não estavam invertidos?
Dizer que foi feita validação cruzada com 10 partições.>>

O próximo experimento objetiva verificar a habilidade do modelo em generalizar para qualquer indivíduo. Sabemos de antemão que os aplicativos geralmente vêm com modelos treinados a partir de dados de participantes que certamente não contemplam a diversidade da população. Isso leva a dificuldade em generalizar os modelos e uma acurácia baixa em indivíduos não contemplados no treinamento. Esse problema é agravado quando é considerada a intensidade (leve, moderada e vigorosa) dos movimentos durante a atividade, pois em vez de inferir uma classe por atividade, passam a ser três classes por atividade. 

Envio do Relatório Final
12/07/22 a 16/08/22

Inscrição no XXX Seminário de IC
08/08/22 a 08/09/22

XXX Seminário de Iniciação Científica
Ocorrerá junto com a semana da Ciência
e Tecnologia. Será divulgado nos canais de comunicação.

;; Reescrever o código da iniciação científica para que fique só
;; meu e fique mais claro também.

(code-c-d "ic" "~/ic/" :anchor)
;; (find-icfile "")
;; (find-icfile "code.py")

(defun c () (interactive) (find-icfile "mycode.py"))
(defun c () (interactive) (find-icfile "SensorData.py"))
(defun i () (interactive) (find-icfile "SensorData.txt"))
(defun ic () (interactive) (find-2a '(find-icfile "SensorData.txt") '(find-icfile "SensorData.py")))
(defun ci () (interactive) (find-2a '(find-icfile "SensorData.py") '(find-icfile "SensorData.txt")))
(defun ep () (interactive) (find-3a '(find-icfile "SensorData.py") '(find-icfile "SensorData.txt") '(find-ebuffer eepith-buffer-name)))


;; Ok não fluiu, mas oque eu iria fazer:
;;  refazer o experimento sem o esqueleto de código do cleir.
;;  utilizar testblocks para documentar o código.
;;  escrever sobre o experimento.

;; (find-fline "~/ic/")
;; (find-fline "~/ic/dataset/data")

;; (find-icfile "mycode.py")

;; 1. Código com cleir
;;   (find-fline "~/personificacao-git/")

;; 2. Código para gerar plot de todos os sinais do dataset
;;    (find-fline "~/notebooks/")

;; 3. Código para gerar o plot com error bar
;;    (find-fline "~/pysrc/confidence_interval.py")
