;; vim: set ft=lisp:

(load "package://baxtereus/baxter.l")

(load "package://handle_moving_object/euslisp/handle_moving_object/reachable-space.l")


(setq *baxter* (instance baxter-robot :init))
(setq *robot* *baxter*)

(setq reachable-space (compute-reachable-space :arm :rarm :draw nil))