from pathlib import Path
import re
import hashlib

root = Path('/Users/todmanlaibaatar/Documents/Codex/2026-09-13/ex')
body = (root / 'work/sector-closure-body.tex').read_text()
pre = r'''\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb,mathtools,amsthm,enumitem}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{remark}\newtheorem{remark}[theorem]{Remark}
\allowdisplaybreaks[2]
\title{Quantitative swing-by on an entire probe sector}
\author{Theory continuation: noiseless training and initialization-controlled leakage}
\date{September 14, 2026}
\begin{document}
\maketitle
'''
(root / 'outputs/SWINGBY_SECTOR_REVERSAL.tex').write_text(pre + body + '\n\\end{document}\n')

src_path = Path('/Users/todmanlaibaatar/Downloads/main-10.tex')
src = src_path.read_text()
main_part, proof_part = body.split('\\section{Proof of the sector theorem}', 1)
proof_part = '\\section{Proof of the sector theorem}' + proof_part
assert src.count('\\bibliography{iclr2027_conference}') == 1
assert src.count('\\end{document}') == 1
new = src.replace('\\bibliography{iclr2027_conference}',
                  main_part + '\n\\bibliography{iclr2027_conference}', 1)
new = new.replace('\\end{document}', proof_part + '\n\\end{document}', 1)
labels = re.findall(r'\\label\{([^}]+)\}', new)
assert len(labels) == len(set(labels)), 'Duplicate labels'
(root / 'outputs/main-11-sector-review.tex').write_text(new)
print('Built standalone proof and integrated review. Source SHA-256:', hashlib.sha256(src_path.read_bytes()).hexdigest())
