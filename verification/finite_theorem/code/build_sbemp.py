from pathlib import Path
import hashlib,difflib
R=Path(__file__).resolve().parents[1]
D=Path('/Users/todmanlaibaatar/Downloads')
expected={'main-14.tex':'11de1e09ba39f91381010926a39426db26b8af7fd98b3b9a6c10defe0a7bd0bf','SWINGBY_THEORY_HANDOFF_v11.md':'9066e82449172b5e73bc6e48fb69b31a5d36c0f1223bc0a50fb8fa4fa7ed1e39'}
for name,digest in expected.items():
 p=D/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, 'Original changed; reconcile before writing'
 backup=R/'outputs'/(p.stem+'-before-empirical-two-mean'+p.suffix)
 if backup.exists():assert backup.read_bytes()==p.read_bytes()
 else:backup.write_bytes(p.read_bytes())
base=(D/'main-14.tex').read_text()
assert r'\label{sbemp:scope}' not in base
head,tail=base.split(r'\section{The default feedback channel: coupled ratios and fixed gate endpoints}',1)
tail=tail.replace(r'N_W\le2\gamma_W',r'N_W\le\sqrt{2}\gamma_W')
tail=tail.replace(r'\frac{2\gamma_W R_W}{\ell}',r'\frac{\sqrt{2}\gamma_W R_W}{\ell}')
anchor='Every remaining neuron belongs to an explicit exception sum in $N$.'
assert tail.count(anchor)==1
tail=tail.replace(anchor,
 r'Indeed, balance gives $\sum_Wz_{i2}^2\le S_W$, while'+ '\n'+
 r'$\sum_Wv_{i1}^2\le2T_W$; Cauchy--Schwarz yields $\sqrt{2}$.'+'\n'+anchor)
anchor='still require a common initialized-trajectory certificate.'
assert tail.count(anchor)==1
tail=tail.replace(anchor,anchor+'\n'+r'Section~\ref{sbemp:scope} strengthens holding by retaining empirical'+ '\n'+r'moments in the reference field and tracking both weighted tilt means.')
body=(R/'work/sbemp-continuation-body.tex').read_text()
new=head+r'\section{The default feedback channel: coupled ratios and fixed gate endpoints}'+tail
new=new.replace(r'\end{document}',body+'\n'+r'\end{document}')
(R/'outputs/main-14-empirical-two-mean-staged.tex').write_text(new)
(R/'outputs/SWINGBY_EMPIRICAL_TWO_MEAN_MAIN14.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='main-14-before.tex',tofile='main-14.tex')))
h=(D/'SWINGBY_THEORY_HANDOFF_v11.md').read_text()
anchor='## September 16 continuation: coupled-ratio delivery of the default c21 channel'
assert h.count(anchor)==1
h=h.replace(anchor,(R/'work/sbemp-handoff-update.md').read_text()+'\n'+anchor)
h=h.replace('The default post-B1 route is `c_{21}^O`-only. The `c_{12}^O` coherence package is optional unless its finite-sample transfer becomes much sharper.',
 'The current post-B1 priority is empirical two-mean holding (`sbemp:*`): entry, the exact error bounds, and timing. This directly grades the strong block. The `c_{21}^O` certificate remains available when needed for the full-network signed gap; `c_{12}^O` coherence is optional. Complement grading is not automatic.')
(R/'outputs/SWINGBY_THEORY_HANDOFF_v11-empirical-two-mean-staged.md').write_text(h)
preamble=(R/'outputs/SWINGBY_C21_COUPLED_CONTINUATION.tex').read_text().split(r'\begin{document}')[0]
preamble=preamble.replace('Swing-by: coupled-ratio delivery of the default feedback channel','Swing-by: empirical holding with two tracked means')
(R/'outputs/SWINGBY_EMPIRICAL_TWO_MEAN_CONTINUATION.tex').write_text(preamble+'\\begin{document}\n\\maketitle\n'+body+'\n\\end{document}\n')
(R/'work/sbemp-build').mkdir(exist_ok=True)
print('Backups and staged manuscript/handoff prepared.')
