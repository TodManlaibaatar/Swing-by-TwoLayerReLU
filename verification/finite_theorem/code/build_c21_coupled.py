from pathlib import Path
import hashlib

root=Path(__file__).resolve().parents[1]
downloads=Path("/Users/todmanlaibaatar/Downloads")
main=downloads/"main-14.tex"
handoff=downloads/"SWINGBY_THEORY_HANDOFF_v11.md"
expected={
 "main-14.tex":"7b7c46e1b0dbffd401e03af32b0bdd23a450c16959bcd837a00ce5ca10c1e0c1",
 "SWINGBY_THEORY_HANDOFF_v11.md":"5777a64b313993491288ce026307c40c35257c5eb7c10b773a0ec95c2118d61f",
}
for p in [main,handoff]:
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[p.name], (
        "Source changed; reconcile before writing",p)
    backup=root/"outputs"/(p.stem+"-before-c21-coupled"+p.suffix)
    if backup.exists(): assert backup.read_bytes()==p.read_bytes()
    else: backup.write_bytes(p.read_bytes())

body=(root/"work/c21-coupled-continuation-body.tex").read_text()
base=main.read_text()
assert base.count("\\end{document}")==1 and "\\label{sbpos:" not in base
old="first-hitting creation interface.\n\\end{remark}"
assert base.count(old)==1
base=base.replace(old,
    "first-hitting creation interface. The coupled-ratio continuation in\n"
    "Section~\\ref{sbpos:scope} supplies an alternative feedback certificate,\n"
    "and Theorem~\\ref{sbpos:aggregate-mean} proves a population holding\n"
    "region using the weighted output-tilt mean. Its initialized empirical\n"
    "entry and transfer remain open.\n\\end{remark}")
staged=base.replace("\\end{document}",
    "\n% September 16: coupled c21 continuation; general theorem remains open.\n"
    +body+"\n\\end{document}")
(root/"outputs/main-14-c21-staged.tex").write_text(staged)

h=handoff.read_text().replace("main-13","main-14")
h=h.replace("**Revision date:** 2026-09-15", "**Revision date:** 2026-09-16")
h=h.replace("Supersedes v9 and earlier handoffs.","Supersedes v10 and earlier handoffs.")
anchor="## September 15: direct signed-gap and coherent-tilt continuation"
idx=h.index(anchor)
intro=(root/"work/c21-handoff-update.md").read_text()
h=h[:idx]+intro+"\n"+h[idx:]
h=h.replace("and the new `sbcoh:*` post-B1 strong-tilt coherence / weak-rotation module.",
    "the `sbcoh:*` post-B1 strong-tilt coherence / weak-rotation module, and the September 16 `sbpos:*` coupled-ratio, fixed-endpoint, and aggregate-mean continuation.")
h=h.replace("This is the highest-value next lemma.",
    "**September 16 update:** The conditional feedback delivery is now installed in `sbpos:feedback`, with coupled entry and the population aggregate-mean holding theorem `sbpos:aggregate-mean`. The remaining target is to deliver their hypotheses on the initialized empirical trajectory. The older polarization route below remains an alternative; its transverse matching is not compulsory for the new coupled-ratio route.")
h=h.replace("3. **Output/input matching:** turn B1/T1.7 mismatch information into `a_z,b_z,d_*`.",
    "3. **Coupled transfer:** for the new route, bound the normalized input, output and aggregate-mean errors on the reached rectangle. The older polarization route instead requires turning B1/T1.7 mismatch information into `a_z,b_z,d_*`; do not impose that extra transverse matching on `sbpos` automatically.")
(root/"outputs/SWINGBY_THEORY_HANDOFF_v11-c21-staged.md").write_text(h)

preamble=r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage[colorlinks=true]{hyperref}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\title{Swing-by: coupled-ratio delivery of the default feedback channel}
\author{Theory continuation for review}
\date{September 16, 2026}
\begin{document}
\maketitle
"""
(root/"outputs/SWINGBY_C21_COUPLED_CONTINUATION.tex").write_text(
    preamble+body+"\n\\end{document}\n")
(root/"work/c21-coupled-build").mkdir(exist_ok=True)
print("Backups, staged main-14/handoff, and standalone proof prepared.")
