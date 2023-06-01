;; This file is used for logging what I am doing between commits.

(code-c-d "twexp" "~/ic/src/experiment_time_warping/")
(code-c-d "daily" "~/org-roam/daily/")
(code-c-d "ic" "~/ic/")

;; Agora estou escrevendo sobre o experimento, e gerando um pdf..
;  (find-icfile "documents/data_augmentation.org")

;; Quando li e escreví sobre aumento de dados.
;  (find-fline "/home/brnm/org-roam/daily/2022-09-29.org")
;  (find-fline "/home/brnm/org-roam/daily/2022-09-29.org" "* Lendo artigo de <<<data augmentation>>>")

; !!!! Tenho que entender como que põe citação pelo emacs.. no orgmode, sei que tem como..


;; 1. Ler oque escrevi sobre aumento de dados

;; Add bibtex to pdf dir
; (find-fline "~/pdfs/ic-har/")


;; File used for testing
;  (find-twexpfile "plotting_given_arpos.py")
;  (find-twexpfile "plotting_given_arpos.py" "ap.plot_compare_err_bar(")
;;
;; Function to execute the test
(defun doit (file_name) (interactive "sFile name:")
       (shell-command
       (concat "cd ~/ic/src/experiment_time_warping/; python plotting_given_arpos.py " file_name))
       (if (string= file_name "")
          (find-file "~/ic/src/experiment_time_warping/default_name_2plots.png")
        (find-file (concat "~/ic/src/experiment_time_warping/" file_name))))

;  (find-icfile "src/arpc_plot.py" "def plot_compare_2_set_of_exps(")
;  (find-icfile "src/arpc_plot.py" "def fix_text_for_xlabel_caption")
