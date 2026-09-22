from pathlib import Path
import hashlib,difflib,re
R=Path.cwd();D=Path('/Users/todmanlaibaatar/Downloads')
expected={'main-14.tex':'57667b363acf8ef39a1c9a345fd7137638444808731e20622431f78119a6d3e0','SWINGBY_THEORY_HANDOFF_v12.md':'0aca3b046010c2d8c4abd69976c5de9115ee705d43d78d2ff326108cb91407cd'}
for name,h in expected.items():
 p=D/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,'Source changed; reconcile first'
 backup=R/'outputs'/(p.stem+'-before-finite-window'+p.suffix)
 if backup.exists():assert backup.read_bytes()==p.read_bytes()
 else:backup.write_bytes(p.read_bytes())
body=(R/'work/sbwindow-continuation-body.tex').read_text();base=(D/'main-14.tex').read_text()
assert r'\label{sbwindow:scope}' not in base
old='''Its larger eigenvalue is approximately $-0.42511$. Replacing the retained
residual damping by the loss-norm coefficient $K$ instead gives a larger
eigenvalue approximately $3.99817$. Thus the coupled calculation can
improve qualitatively on the coarse mismatch Gronwall even with nonzero
excluded mass. This is a frozen-coefficient algebraic check, not a
trajectory, a sign-invariance proof, or an initialized entry certificate.
Gate changes, noise forcing, mixed quadrants and time-varying coefficients
must still be paid in \\eqref{sbcouple:solution}.'''
replacement=r'''Its larger eigenvalue is approximately $-0.42511$. This demonstrates the
qualitative benefit of retained damping for this mass-relative
\emph{matched-sector two-variable subsystem}, with $R_J=0$.
Replacing the residual damping by the loss-norm coefficient $K$ instead
gives a larger eigenvalue approximately $3.99817$, destroying even that
matched-sector stabilization.

Neither number implies contraction of the full physical-time
three-variable comparison. Its matrix $C$ is Metzler, so
$s(C)\ge\max_i C_{ii}\ge\theta_J$: shift $C$ to a nonnegative matrix
and use that its spectral radius dominates each diagonal entry.
Consequently a positive $\theta_J$ precludes asymptotic contraction of
that comparison. Time-dependent mass normalization changes its diagonal
and requires a separate estimate. The finite-window theorem needs an
affordable Duhamel budget, not asymptotic contraction. This is a
frozen-coefficient algebraic check, not a trajectory, sign-invariance or
initialized entry certificate. Gate changes, noisy forcing, mixed
quadrants and changing coefficients must still be paid in
\eqref{sbcouple:solution}; Section~\ref{sbwindow:scope} begins that
finite-window calculation from initialization.'''
assert base.count(old)==1
new=base.replace(old,replacement).replace(r'\end{document}',body+'\n'+r'\end{document}')
old2='The first remaining primitive-parameter task is now quantitative and\ncontains no hidden'
new2='The next display records the entry budget at this stage of the proof.\nSection~\\ref{sbwindow:scope} supplies a from-zero sufficient calculation\nand its explicit timing premises; the downstream relative feedback\nfrontier is \\eqref{sbwindow:next-gap}. The entry budget\ncontains no hidden'
assert new.count(old2)==1;new=new.replace(old2,new2)
labels=re.findall(r'\\label\{([^}]+)\}',body);alllabels=re.findall(r'\\label\{([^}]+)\}',new)
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',body)
assert len(labels)==len(set(labels)) and all(alllabels.count(x)==1 for x in labels)
assert not set(refs)-set(alllabels),set(refs)-set(alllabels)
(R/'outputs/main-14-finite-window-staged.tex').write_text(new)
(R/'outputs/SWINGBY_FINITE_WINDOW_MAIN14.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='main-14-before.tex',tofile='main-14.tex')))
hbase=(D/'SWINGBY_THEORY_HANDOFF_v12.md').read_text();anchor='## September 16 coupled signed-source continuation — current frontier'
assert hbase.count(anchor)==1
handoff=hbase.replace(anchor,(R/'work/sbwindow-handoff-update.md').read_text()+'\n## Earlier September 16 coupled signed-source continuation — frontier superseded above')
handoff=handoff.replace('This is not a reached-trajectory or parameter-region certificate.','This check assumes R_J=0 and concerns only the matched-sector two-variable system. The full physical three-variable Metzler comparison has spectral abscissa at least theta_J; positive theta_J prevents its asymptotic contraction. Neither check is a reached-trajectory or parameter-region certificate.',1)
handoff=handoff.replace('Earlier review-copy path names are provenance only.','The `sbcouple:*` and `sbwindow:*` sections add signed complement propagation, finite-window from-zero entry, a signed negative-coordinate repair yielding positive c21, and the explicitly unclosed full-network grading gap. Earlier review-copy path names are provenance only.',1)
(R/'outputs/SWINGBY_THEORY_HANDOFF_v12-finite-window-staged.md').write_text(handoff)
(R/'outputs/SWINGBY_FINITE_WINDOW_HANDOFF.patch').write_text(''.join(difflib.unified_diff(hbase.splitlines(True),handoff.splitlines(True),fromfile='handoff-before.md',tofile='handoff.md')))
preamble=(R/'outputs/SWINGBY_COUPLED_SIGNED_CONTINUATION.tex').read_text().split('\\makeatletter')[0]
preamble=preamble.replace('signed cohort sources and coupled complement propagation','from-zero finite-window propagation and entry')
external=sorted(set(refs)-set(labels));stubs='\\makeatletter\n'+''.join('\\@namedef{r@'+x+'}{{source}{}}\n' for x in external)+'\\makeatother\n'
(R/'outputs/SWINGBY_FINITE_WINDOW_CONTINUATION.tex').write_text(preamble+stubs+'\\begin{document}\n\\maketitle\n'+body+'\n\\end{document}\n')
(R/'work/sbwindow-build').mkdir(exist_ok=True)
print('Prepared manuscript, handoff, backups, diffs and standalone. New labels:',len(labels),'All new references resolve.')
