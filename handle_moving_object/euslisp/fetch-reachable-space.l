;; vim: set ft=lisp:

(load "package://fetcheus/fetch.l")

(setq *fetch* (instance fetch-robot :init))
(setq *robot* *fetch*)


(defun compute-main ()
  (setq reachable-space (compute-reachable-space :draw nil))
  (visualize-reachable-space reachable-space)
  (dump-loadable-structure "data/fetch-reachable-space.l" reachable-space)
  )


(defun load-main ()
  (load "package://handle_moving_object/data/fetch-reachable-space.l")
  (visualize-reachable-space reachable-space)
  )


;; ---------------------------------------
;; main
;; ---------------------------------------
(load-main)
