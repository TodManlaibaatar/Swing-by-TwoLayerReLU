from pathlib import Path
import hashlib,difflib
R=Path(__file__).resolve().parents[1];D=Path('/Users/todmanlaibaatar/Downloads')
expected={'main-14.tex':'4d396b9c6ebbfd03cd74d5aa96fb755eb9436deb0b40d88114ae075b8811f844','SWINGBY_THEORY_HANDOFF_v12.md':'445cc3f4625ee6ebb4ef77ef7418134b9321cc6ccf8c313774aafd8e77a9b403'}
for name,digest in expected.items():
 p=D/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,'Source changed; reconcile before applying'
 backup=R/'outputs'/(p.stem+'-before-weighted-transport'+p.suffix)
 if backup.exists():assert backup.read_bytes()==p.read_bytes()
 else:backup.write_bytes(p.read_bytes())
base=(D/'main-14.tex').read_text();new=base
assert r'\label{sbflux:scope}' not in base
new=new.replace(r'Therefore $D^+[-z_s]_+\le-\Gamma_u(\tau)$, and',r'While $z_s<0$, $D^+[-z_s]_+\le-\Gamma_u(\tau)$, and')
new=new.replace(r'Thus $D^+[-z_t]_+\le-\Gamma_w(\tau)$.',r'While $z_t<0$, $D^+[-z_t]_+\le-\Gamma_w(\tau)$.')
anchor=r'\begin{remark}[Canonical crossing margins]'
assert new.count(anchor)==1
note=r'''\begin{remark}[The pre-crossing output lower bound is a separate guard]
T1.9 alone gives $t_i\ge-T(\tau)$ before input crossing. The smaller
$-\tan(D_{\rm lock})/e$ bound follows from its angle-gap estimate only
after input nonnegativity. Lemma~\ref{sbflux:pre-cross-output} derives a
concurrent output lower envelope and retains favorable forcing through
\eqref{sbflux:reflected-output}; use that envelope in the input margin
unless the stronger pre-crossing bound has been proved separately.
\end{remark}

'''
new=new.replace(anchor,note+anchor)
body=(R/'work/sbflux-continuation-body.tex').read_text()
new=new.replace(r'\end{document}',body+'\n'+r'\end{document}')
(R/'outputs/main-14-weighted-transport-staged.tex').write_text(new)
(R/'outputs/SWINGBY_WEIGHTED_TRANSPORT_MAIN14.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='main-14-before.tex',tofile='main-14.tex')))
h=(D/'SWINGBY_THEORY_HANDOFF_v12.md').read_text()
anchor='# SEPTEMBER 16 LATE UPDATE — START HERE: THEORY-CLOSURE MISSION'
assert h.count(anchor)==1
h=h.replace(anchor,(R/'work/sbflux-handoff-update.md').read_text()+'\n'+anchor)
h=h.replace('- for later same-sign entrants, derive a **weighted boundary-flux / angular-transport bound**. Use', '- for later same-sign entrants, the exact fixed-family weighted transport is now installed in `sbflux:transport`. Close its **signed production and handoff coverage bounds** before using it in the crossing clock. The natural weight is')
(R/'outputs/SWINGBY_THEORY_HANDOFF_v12-weighted-transport-staged.md').write_text(h)
preamble=(R/'outputs/SWINGBY_C21_COUPLED_CONTINUATION.tex').read_text().split(r'\begin{document}')[0]
preamble=preamble.replace('Swing-by: coupled-ratio delivery of the default feedback channel','Swing-by: fixed-complement transport and concurrent output bounds')
(R/'outputs/SWINGBY_WEIGHTED_TRANSPORT_CONTINUATION.tex').write_text(preamble+'\\begin{document}\n\\maketitle\n'+body+'\n\\end{document}\n')
(R/'work/sbflux-build').mkdir(exist_ok=True)
print('Backups and staged current manuscript/v12 handoff prepared.')
