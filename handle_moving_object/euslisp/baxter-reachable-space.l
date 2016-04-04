;; vim: set ft=lisp:

(load "package://baxtereus/baxter.l")

(load "package://handle_moving_object/euslisp/handle_moving_object/reachable-space.l")


(defun main ()
  (setq reachable-space (compute-reachable-space :draw nil))
  (visualize-reachable-space reachable-space)
  (dump-loadable-structure "data/baxter-rarm-reachable-space.l" reachable-space)
  )


(setq *baxter* (instance baxter-robot :init))
(setq *robot* *baxter*)
(main)
