;; vim: set ft=lisp:

(load "package://baxtereus/baxter.l")

(load "package://handle_moving_object/euslisp/handle_moving_object/reachable-space.l")


(setq *baxter* (instance baxter-robot :init))
(setq *robot* *baxter*)


(defun compute-main ()
  (setq reachable-space (compute-reachable-space :draw nil))
  (visualize-reachable-space reachable-space)
  (dump-loadable-structure "data/baxter-rarm-reachable-space.l" reachable-space)
  )


(defun load-main ()
  (load "package://handle_moving_object/euslisp/data/baxter-rarm-reachable-space.l")
  (visualize-reachable-space reachable-space)
  )


;; ---------------------------------------
;; main
;; ---------------------------------------
(load-main)
