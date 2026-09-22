from pathlib import Path
import re
root=Path('/Users/todmanlaibaatar/Documents/Codex/2026-09-13/ex')
body=(root/'work/signed-body.tex').read_text()
pre=r'''\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb,mathtools,amsthm,enumitem}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{remark}\newtheorem{remark}[theorem]{Remark}
\allowdisplaybreaks[2]
\title{Swing-by: signed propagation and a fixed-gate reversal theorem}
\author{Proof continuation for the main-9 review}
\date{September 14, 2026}
\begin{document}
\maketitle
'''
(root/'outputs/SWINGBY_SIGNED_PROPAGATION.tex').write_text(pre+body+'\n\\end{document}\n')
for src,dest in [('outputs/main-11-review.tex','outputs/main-12-review.tex'),
                 ('work/aggregate-appendix-check.tex','work/signed-appendix-check.tex')]:
    old=(root/src).read_text()
    assert old.count('\\end{document}')==1
    new=old.replace('\\end{document}',body+'\n\\end{document}')
    labels=re.findall(r'\\label\{([^}]+)\}',new)
    assert len(labels)==len(set(labels))
    (root/dest).write_text(new)
print('Installed signed propagation and reversal proofs; labels unique.')
