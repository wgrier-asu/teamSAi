;; Do not reformat this directory
((python-mode . ((eval . (progn
                           (conda-env-autoactivate-mode)
                           (remove-hook 'before-save-hook #'adria-python-format-buffer nil t)))))
 (js-json-mode . ((eval . (progn
                            (adria-json-on-save-mode 0))))))
