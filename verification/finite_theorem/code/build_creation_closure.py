from pathlib import Path
import re, hashlib

root=Path('/Users/todmanlaibaatar/Documents/Codex/2026-09-13/ex')
body=(root/'work/creation-closure-body.tex').read_text()
pre=r'''\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb,mathtools,amsthm,enumitem}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\allowdisplaybreaks[2]
\title{Creation certificates for the general Swing-by model}
\author{Whole-network regulation, transition-kernel positivity, and joint concentration}
\date{September 15, 2026}
\begin{document}
\maketitle
'''
(root/'outputs/SWINGBY_CREATION_CLOSURE.tex').write_text(pre+body+'\n\\end{document}\n')
src=root/'outputs/main-11-sector-review.tex'
old=src.read_text()
assert old.count('\\end{document}')==1
def once(a,b):
    global old
    assert old.count(a)==1, a[:80]
    old=old.replace(a,b,1)
once(r'''A predecessor
positive-output generation lemma must combine a positive-coordinate cohort
with \eqref{eq:beff-exact} and charge the complementary negative-output mass.''',
r'''One possible predecessor route combines a positive-coordinate cohort
with \eqref{eq:beff-exact} and charges the complementary negative-output mass.
An alternative is the exact whole-network regulator in
Proposition~\ref{sbgen:network-regulator} and
Corollary~\ref{sbgen:deficit-certificate}; that route instead controls
its signed directional forcing and accumulated damping clock.''')
once(r'''Any generation lemma used to discharge
\eqref{eq:GC-reg} must therefore combine
\eqref{eq:block-b-lower} with a one-sided bridge from the positive cohort to
\eqref{eq:beff-exact} and an explicit bound on $\mathcal N_b$ (or a sharper
signed substitute).''',
r'''For this cohort-based route, discharging \eqref{eq:GC-reg} requires
combining \eqref{eq:block-b-lower} with a one-sided bridge to
\eqref{eq:beff-exact} and a bound on $\mathcal N_b$ or a sharper signed
substitute. The whole-network route in
Corollary~\ref{sbgen:deficit-certificate} avoids a separate cohort bridge;
its forcing bound is a different, still open dynamical obligation.''')
once(r'''This preserves the covariance cancellation that is lost by charging the two
ratios independently.''',
r'''This preserves the covariance cancellation that is lost by charging the two
ratios independently. Proposition~\ref{sbgen:joint-gap} now supplies an
explicit joint influence bound with the ratio remainder; use
Corollary~\ref{sbgen:gap-clipping} if the population gap is unclipped.''')
once(r'''No global analytic sign theorem over all SIM parameters
is assumed.''',
r'''No global analytic sign theorem over all SIM parameters
is assumed. Theorem~\ref{sbgen:J-positive} supplies an explicit analytic
high-SNR subregion, and Corollary~\ref{sbgen:J-finite} gives its
finite-sample transfer. Their conservative bounds do not certify the
canonical noise level by themselves.''')
new=old.replace('\\end{document}',body+'\n\\end{document}',1)
labels=re.findall(r'\\label\{([^}]+)\}',new)
assert len(labels)==len(set(labels)), 'duplicate labels'
(root/'outputs/main-12-creation-review.tex').write_text(new)
print('Built standalone and integrated continuation; base:',hashlib.sha256(src.read_bytes()).hexdigest())
