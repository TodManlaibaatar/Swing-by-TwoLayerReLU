from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
source = Path("/Users/todmanlaibaatar/Downloads/main-12.tex")
base = source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "6b26f8778618fd6ca08db1ed4227cc69b2f00a6553abbc7887e25f70e90cab74"
)
body = (root/"work/signed-gap-continuation-body.tex").read_text()
preamble = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage[colorlinks=true]{hyperref}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\title{Swing-by: signed-gap and generation continuation}
\author{Proof continuation for review}
\date{September 15, 2026}
\begin{document}
\maketitle
"""
(root/"outputs/SWINGBY_SIGNED_GAP_CONTINUATION.tex").write_text(
    preamble+body+"\n\\end{document}\n"
)
assert base.count("\\end{document}")==1
assert "\\label{sbgap:" not in base
review=base.replace("\\end{document}",
    "\n% September 15 signed-gap continuation; general theorem still open.\n"
    +body+"\n\\end{document}")
(root/"outputs/main-13-signed-gap-review.tex").write_text(review)
print("Built standalone proof and additive main-13 review copy.")
