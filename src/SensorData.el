(code-c-d "ic" "~/ic/" :anchor)

(defun ep ()
  (interactive)
  (find-3b
   '(find-icfile "SensorData.py")
   '(find-icfile "SensorData.txt")
   '(find-ebuffer eepitch-buffer-name)))

;; Mudanças no dataset
;;  Remover primeira série temporal - atividade andando - Aluno0
;;  Remover primeira séria temporal - Deitado vigoroso  - Aluno7
;;  Remover primeira séria temporal - Deitado vigoroso  - Aluno8
;;  [Feito]  Remover primeira série temporal de todos os replicados??
;;                    (find-icfile "SensorData.py" "if remFirst:")
;;
;;  Rotacionar dado de DeitadoVigoroso - Aluno1  
;;  Rotacionar todos os Deitado        - Aluno 4
;;  Rotacionar todos os deitado        - Aluno 7
;;  Pergunta: Os dados rotacionados vão mesmo interferir no modelo?
;;             O modelo não consegue abstrair a orientação dos dados?
;;  [Feito] : Função que rotaciona uma classe
;                  (find-icfile "SensorData.py" "def rotate_class")
;;
;; TODO: Checar se existem duplicatas nos dados [Automatizado]
;;
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
 (ep)
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.remove_beginning()
sd.fix_dup(remFirst=True)
sd.plot_all(participantes=sd.participantes[1:2])


;; Rotacionando os dados da uma classe, simulando o celular estando em
;; outra posição
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
 (ep)
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.remove_beginning()
sd.rotate_class([(1., 'Deitado', 'Moderado')], [0,0,1])
sd.plot_all(participantes=[1])

 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
 (ep)
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.remove_beginning()
sd.fix_dup(remFirst=True)
classes = [(1., 'Deitado', 'Moderado')]
classes += [(4., 'Deitado', i) for i in ['Leve', 'Moderado', 'Vigoroso']]
classes += [(7., 'Deitado', i) for i in ['Leve', 'Moderado', 'Vigoroso']]
classes
sd.rotate_class(classes, [0,0,1])
sd.plot_all()

;; Estou desenvolvendo um método que:
;;  Encontra o indice onde o tempo de uma amostra é menor que o tempo da amostra anterior
;;  Deste indice para frente soma o tempo da amostra anterior
;; Desta forma uma série duplicada passa a ser uma série mais longa
 
;            (find-icfile "SensorData.py" "def fix_dup(self):")

;; Este método erra quando se remove os 10 primeiros segundos dos dados
;;  O erro é porque quando removo os 10 segundos iniciais removo todos as
;;  amostras que possuem timestamp menor que 10000, para as classes com
;;  dados duplicados, significa que removo o início da segunda série temporal
;;  então há uma lacuna de indices no meio dos dados da classe.
;;
;;         (find-icfile "SensorData.py" "df = df.reset_index()")
;;
;; 
;; Depois de resolver esse problema, agora há um gap entre o tempo final
;; de uma série temporal e outra, no gráfico aparece uma reta entre o fim de
;; uma série e o início de outra.
;;
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
 (ep)
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.remove_beginning()
sd.fix_dup()
sd.plot_all()




 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
 (ep)
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.remove_outliers()
sd.remove_beginning()
sd.fix_dup(remFirst=True)
classes = [(1., 'Deitado', 'Moderado')]
classes += [(4., 'Deitado', i) for i in ['Leve', 'Moderado', 'Vigoroso']]
classes += [(7., 'Deitado', i) for i in ['Leve', 'Moderado', 'Vigoroso']]
sd.rotate_class(classes, [0,0,1])
sd.segment_extract_features()
sd.confusion_matrix()
sd.kfold_crossval(10)
