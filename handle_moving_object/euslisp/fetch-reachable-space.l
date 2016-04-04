;; vim: set ft=lisp:

(load "package://fetcheus/fetch.l")

(load "package://handle_moving_object/euslisp/handle_moving_object/reachable-space.l")


(defun main ()
  (setq reachable-space (compute-reachable-space :draw nil))
  (visualize-reachable-space reachable-space)
  (dump-loadable-structure "data/fetch-reachable-space.l" reachable-space)
  )


(setq *fetch* (instance fetch-robot :init))
(setq *robot* *fetch*)
(main)
