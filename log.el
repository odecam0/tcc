;; This file is used for logging what I am doing between commits.

(code-c-d "twexp" "~/ic/src/experiment_time_warping/")

;; Now I am doing the caption of 
;  (find-icfile "src/arpc_plot.py" "def plot_compare_err_bar(")
;  (find-icfile "src/arpc_plot.py" "def plot_compare_err_bar(" "if caption:")
;  DONE!


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


;; Have to do the same for
;  (find-icfile "src/arpc_plot.py" "def plot_compare_2_set_of_exps(")

;; The thing with this one, is that we have to put something more in the caption, adding to the
;; previously done caption, so we have to find some mechanism to do this.
;;
;; The caption is done in the previous one by seting the x_label.
;  (find-icfile "src/arpc_plot.py" "ax.set_xlabel(text, loc='left')")
;; So if we have acess to the same axis that we called when using set_xlabel, we might be able to
;; do this.
