;; ----------------------------------------------------------
;; ----------- Sobre iniciação científica -------------------


;; https://mail.google.com/mail/u/0/#search/alessandro_copetti%40id.uff.br

;; Envio do Relatório Final
;; 12/07/22 a 16/08/22

;; Inscrição no XXX Seminário de IC
;; 08/08/22 a 08/09/22

;; XXX Seminário de Iniciação Científica
;; Ocorrerá junto com a semana da Ciência
;; e Tecnologia. Será divulgado nos canais de comunicação.

 ISSO TUDO JÁ FOI FEITO

(code-c-d "ic" "~/ic/" :anchor)
;; (find-fline "~/ic/")
;; (find-fline "~/ic/dataset/data")
;; (find-icfile "src/mycode.py")

(defun in () (interactive) (find-icfile "src/ic.el"))

(defun co () (interactive) (find-icfile "src/mycode.py"))
;; (defun co () (interactive) (find-icfile "src/mycode.py" "def train_model_split1"))

(defun c () (interactive) (find-icfile "src/SensorData.py"))
(defun i () (interactive) (find-icfile "src/SensorData.el"))
(defun ic () (interactive) (find-2a '(i) '(c)))
(defun ci () (interactive) (find-2a '(c) '(i)))
(defun ep () (interactive) (find-3a '(c) '(i) '(find-ebuffer eepith-buffer-name)))


(defun tw () (interactive) (find-icfile "src/TimeWarpWindow.py"))
;; 1. Código com cleir
;;   (find-fline "~/personificacao-git/")

;; 2. Código para gerar plot de todos os sinais do dataset
;;    (find-fline "~/notebooks/")

;; 3. Código para gerar o plot com error bar
;;    (find-fline "~/pysrc/confidence_interval.py")
