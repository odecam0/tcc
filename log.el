;; This file is used for logging what I am doing between commits.

(code-c-d "twexp" "~/ic/src/experiment_time_warping/")

;  (find-icfile "src/arpc_plot.py" "def plot_compare_err_bar")
;  (find-icfile "src/arpc_plot.py" "def plot_compare_2_set_of_exps")
;;
;  (find-twexpfile "")

;; For some reason the plot from plot_compare_2_set_of_exps has 2 things wrong.
;; 1. The second plot hides the first plot
;; 2. The plot with augmented data is in wrong order (more complicated)

;  (find-twexpfile "plotting_given_arpos.py" "test = load_group_of_experiments(test, 'all_exps_arpos')")
;;     in this file /\ I load an object stored in a file using the pickle python module,
;;     which has a list of models and confusion matrixes for 6 different experiments.
;;
;;                This \/ is the pickled file.
;  (find-twexpfile "" "all_exps_arpos")
;;
;; The following function executes the previous file and access the plot image created.
(defun doit (file_name) (interactive "sFile name:")
       (shell-command
	(concat "cd ~/ic/src/experiment_time_warping/; python plotting_given_arpos.py " file_name))
       (if (string= file_name "")
	   (find-file "~/ic/src/experiment_time_warping/default_name_2plots.png")
	 (find-file (concat "~/ic/src/experiment_time_warping/" file_name))))
;;
;; The code that creates the file with the pickled experiments is here \/
;  (find-twexpfile "exp_time_w.py" "save_group_of_experiments(test, 'all_exps_arpos')")
;  (find-twexpfile "exp_time_w.py" "def save_group_of_experiments")
;;
;; The code that implements the plotting funcion
;  (find-icfile "src/arpc_plot.py" "def plot_compare_2_set_of_exps")
;  (find-icfile "src/arpc_plot.py" "def plot_compare_err_bar")
;  (find-icfile "src/arpc_plot.py" "def get_compare_side_err_barr_data")
;;
;; These 3 images represent the problem in hand.
;  (find-twexpfile "2_plots_uncommented.png")
;  (find-twexpfile "1st_plot_commented.png")
;  (find-twexpfile "2nd_plot_commented.png")
;; The first image is identical to the second image, but it should contain both plots.
;; The second image is right.
;; The third image has the order of experiments within each label wrong.

;; Solving the third image
;; The problem has been reduced to this point
;  (find-twexpfile "plotting_given_arpos.py" "arpo.name = pickled_data[0][0]")
;; For some reason, when building de arpo back, the last peephole is not included. 
;; (doit "2nd_plot_commented.png")
;; FIXED! Typo 'conusion_matrixes' instead of 'confusion_matrixes'.
;  (find-twexpfile "plotting_given_arpos.py" "arpo.confusion_matrixes")

;; Solving first image
;  (find-icfile "src/arpc_plot.py" "def plot_compare_err_bar")
;  (find-icfile "src/arpc_plot.py" "sub_gs = gs[0, i].")
;  (eww "GridSpec subgridspec")
;  (eww "getting  GridSpec from figure matplotlib")
;; FIXED! The second plot should have a transparente background
