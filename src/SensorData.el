«.fix_dup»	        (to "fix_dup")
«.rotate_class»	        (to "rotate_class")
«.full_run»	        (to "full_run")
«.error_bar»	        (to "error_bar")
«.scale»	        (to "scale")
«.save_all_data»	(to "save_all_data")
«.time_warp»	        (to "time_warp")
  «.teste wf»	        (to "teste wf")

(code-c-d "ic" "~/ic/" :anchor)

(defun ep ()
  (interactive)
  (find-3b
   '(find-icfile "src/SensorData.py")
   '(find-icfile "src/SensorData.el")
   '(find-ebuffer eepitch-buffer-name)))

«rotate_class»  (to ".rotate_class")
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
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.remove_beginning()
sd.fix_dup(remFirst=True)
sd.plot_all(participantes=sd.participantes[1:2])


;; Rotacionando os dados da uma classe, simulando o celular estando em
;; outra posição
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.remove_beginning()
sd.rotate_class([(1., 'Deitado', 'Moderado')], [0,0,1])
sd.plot_all(participantes=[0])

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

«fix_dup»  (to ".fix_dup")
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
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.remove_beginning()
sd.fix_dup()
sd.plot_all()

«full_run»  (to ".full_run")
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

«error_bar»  (to ".error_bar")
;; Adding the other experiments
; (find-icfile "src/SensorData.py" "def train_model_split1")
; (find-icfile "src/SensorData.py" "def p_gen")
; (find-icfile "src/SensorData.py" "def initial_fix")

"""
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.initial_fix()
sd.segment_extract_features()
result = sd.train_model_split1()
result = sd.train_model_split1(participants=[0,1,2])
"""

; (find-icfile "src/SensorData.py" "def train_model_split2")
"""
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.initial_fix()
sd.segment_extract_features()
result = sd.train_model_split2()
result = sd.train_model_split2(select=[0,1,2])
"""

; (find-icfile "src/SensorData.py" "def train_model_split3")
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.initial_fix()
sd.segment_extract_features()
result = sd.train_model_split3()
result = sd.train_model_split3(select=[0,1,2])

; (find-icfile "src/SensorData.py" "def get_all_accuracies")
; (find-icfile "src/SensorData.py" "def plot_error_bar")
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.initial_fix()
sd.segment_extract_features()
results = sd.get_all_accuracies(save=True)

exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.load_all_accuracies_from_file()
sd.plot_error_bar()
sd.all_accuracies

; (find-icfile "src/SensorData.py" "def plot_error_bar" "if save:")
; (find-icfile "src/plot_error_bar.png")
; (find-icfile "")

«scale»  (to ".scale")
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
; (find-icfile "src/SensorData.py" "def scale_data(self):")
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.initial_fix()
sd.segment_extract_features()
sd.scale_data()
sd.data
; results = sd.get_all_accuracies(save=True, file_name='all_acuracies_standard_before_features')
results = sd.get_all_accuracies(save=True, file_name='all_acuracies_standard_after_features')

scaled_data.loc[scaled_data['tempo']!=np.NaN]
scaled_data.dropna()
self.data.loc[:, non_data_columns]

exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.load_all_accuracies_from_file(file_name='all_acuracies_standard_before_features')
sd.load_all_accuracies_from_file(file_name='all_acuracies_standard_after_features')
sd.plot_error_bar()
sd.all_accuracies

;; saving the pre-processed data into files
«save_all_data»  (to ".save_all_data")
; (find-icfile "src/SensorData.py" "def save_all_data(self):")
; (find-icfile "src/new_data/")
https://docs.python.org/3/library/os.html#os.makedirs
exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.initial_fix()
sd.save_all_data()

exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.initial_fix()
sd.scale_data()
sd.save_all_data(root_dir="./new_data_standard_scale/")


;; Realizando time-warp
«time_warp»  (to ".time_warp")
; (find-icfile "src/SensorData.py" "def time_warp_window")
; (find-icfile "src/SensorData.py" "class SensorData:")
; (find-icfile "src/SensorData.py" "def save_all_data")

; (find-icfile "src/")
; (find-icfile "src/processed_data/")

exec(open("SensorData.py").read(), globals())
sd = SensorData('./processed_data/', extension='.csv')
sd.data

;  ENTÃO! O loading dos dados está ruim
;          Os dados salvos não possuem coluna sensor
;          Os dados originais possuiam..

exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.initial_fix()
sd.data

; (find-icfile "src/SensorData.py" "def segment_extract_features")

;  DIFICULDADE : Utilizando o pandas, eu crio um objeto
;                 rolling do dataframe, e automáticamente
;                 gero as métricas.
;
;                 Em qual ponto intermediário eu iria realizar
;                 o time-warping?

exec(open("SensorData.py").read(), globals())
sd = SensorData()
sd.segment_extract_features()

;  Solução:
; O objeto rolling do pandas é um iterable, então é possível
; fazer [i for i in rolling] pra recuperar dataframes que
; representam cada janela.

«teste wf»  (to ".teste wf")

exec(open("SensorData.py").read(), globals())
sd = SensorData()

df = sd.data
df

; (find-icfile "src/SensorData.py" "def segment_extract_features")
; (find-icfile "src/SensorData.py" "def aip_gen")
for df_local in aip_gen(df):
    df_local  = df_local.astype({'tempo': int})
    # rolling_w = df_local[['x', 'y', 'z', 'tempo']].rolling(window=10, center=True,  on='tempo')
    rolling_w = df_local[['x', 'y', 'z', 'tempo']].rolling(window=10, center=True)
    break

rolling_w
[i for i in rolling_w]

exec(open("TimeWarpWindow.py").read(), globals())

windows           = [i                 for i in rolling_w if i.shape[0] == 10]
augmented_windows = [warp_window(i, 2) for i in windows]
; (find-icfile "src/TimeWarpWindow.py" "def warp_window(")

from pprint import pprint
pprint(list(zip(windows, augmented_windows)))

lst = zip(*augmented_windows)
[i for i in list(zip(*augmented_windows))[0] if i.isnull().values.any()]
[i for i in list(zip(*augmented_windows))[0]]

windows[0]
augmented_windows[0]
type(augmented_windows[0])
type(augmented_windows[0][0])
