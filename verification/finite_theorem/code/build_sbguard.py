from pathlib import Path
import hashlib,re,difflib
R=Path.cwd();D=Path('/Users/todmanlaibaatar/Downloads')
expected={'main-14.tex':'053b48fa8878776e648db51a9c97593e1c88dfc4411cf2361692b64d45477b94','SWINGBY_THEORY_HANDOFF_v12.md':'97f988e181e5396a1aecc7ffb0ce91463b5caede16431695de02e31daeee3750'}
for name,h in expected.items():
 p=D/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,'Source changed'
 b=R/'outputs'/(p.stem+'-before-branch-dimension'+p.suffix)
 if b.exists():assert b.read_bytes()==p.read_bytes()
 else:b.write_bytes(p.read_bytes())
base=(D/'main-14.tex').read_text();new=base
old=r'\tau_{\rm hand}'+'\n'+r' <\min\{\tau_{M_+},\tau_{\rm T1.9},\tau_{\rm branch}\}.'
replacement=r'\tau_{\rm hand}'+'\n'+r' <\min\{\tau_{M_+},\tau_{\rm T1.9}\}.'
assert new.count(old)==1;new=new.replace(old,replacement)
new=new.replace('This is currently the principal quantitative risk of the entry route.','The formerly undefined additional branch deadline is resolved in\nSection~\\ref{sbguard:scope}: entry uses these early guards; subsequent\nfeedback must satisfy $t_g<t_{\\rm hold}$, and generation has its own\nreached continuation exit.',1)
old='''The symbol $\\tau_{\\rm branch}$ currently occurs in the manuscript's
combined-clock ordering without a separate stopping-event definition.
If it denotes the strong tube/radius guards already certified through
$T$, the proposition proves the required ordering for those guards.
If it denotes weak retention or a creation-field condition, its distinct
certificate must be supplied; it cannot be inferred by renaming $T$.'''
replacement='''Section~\\ref{sbguard:scope} resolves the formerly undefined branch
symbol into the actual holding and generation exits. The former is
certified on the priced feedback window; the latter still requires a
reached generation interval and its distinct continuation estimates.'''
assert new.count(old)==1;new=new.replace(old,replacement)
# Correct only the optional sufficient ambient inference; preserve the explicit directional hypotheses.
old=r'{2\lambda(1+e^2S_b^2)}.'
assert new.count(old)==1;new=new.replace(old,r'{2\lambda\sqrt h(1+e^2S_b^2)}.')
old=r'Indeed $\|P_i\|\le\lambda\varepsilon^2\sqrt{hs}$ and'
assert new.count(old)==1;new=new.replace(old,r'Indeed $\|P_i\|\le\lambda s_0\sqrt{s}$ by the aggregate bound and')
old=r'$2\lambda(1+e^2S_b^2)\sqrt{s_0s}/(e\mu_2^2\tau_d)$.'
assert new.count(old)==1;new=new.replace(old,r'$2\lambda\sqrt h(1+e^2S_b^2)\sqrt{s_0s}/(e\mu_2^2\tau_d)$.')
# Expose old feedback dimension dependence where it is used.
anchor=r'\label{sbwindow:positive-feedback}'
new=new.replace(anchor,anchor+'\n'+r'This unprojected version has an explicit dimension-dependent feedback time through $m_0$; see \eqref{sbguard:old-time}--\eqref{sbguard:old-rho}. Corollary~\ref{sbguard:projected-feedback} gives a sharper projected-energy version.'+'\n',1)
anchor=r'\label{sbwindow:entry}'
new=new.replace(anchor,anchor+'\n'+r'The numerical duration below starts at a certified broad-tube time. It is not a dimension-uniform bound on the time from initialization to signed feedback.'+'\n',1)
body=(R/'work/sbguard-continuation-body.tex').read_text();new=new.replace(r'\end{document}',body+'\n'+r'\end{document}')
labels=re.findall(r'\\label\{([^}]+)\}',body);alllabels=re.findall(r'\\label\{([^}]+)\}',new);refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',body)
assert len(labels)==len(set(labels)) and all(alllabels.count(k)==1 for k in labels)
assert not set(refs)-set(alllabels),set(refs)-set(alllabels)
(R/'outputs/main-14-branch-dimension-staged.tex').write_text(new)
(R/'outputs/SWINGBY_BRANCH_DIMENSION_MAIN14.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='main-14-before.tex',tofile='main-14.tex')))
hbase=(D/'SWINGBY_THEORY_HANDOFF_v12.md').read_text();anchor='## September 16 from-zero finite-window continuation — current frontier'
assert hbase.count(anchor)==1
hnew=hbase.replace(anchor,(R/'work/sbguard-handoff-update.md').read_text()+'\n## Earlier September 16 from-zero finite-window continuation — frontier superseded above')
hnew=hnew.replace('The `sbcouple:*` and `sbwindow:*` sections add', 'The `sbguard:*` section audits actual continuation exits and dimension scaling, proves projected-energy and direct holding refinements, and records the remaining late error/common-region gate. The `sbcouple:*` and `sbwindow:*` sections add',1)
(R/'outputs/SWINGBY_THEORY_HANDOFF_v12-branch-dimension-staged.md').write_text(hnew)
(R/'outputs/SWINGBY_BRANCH_DIMENSION_HANDOFF.patch').write_text(''.join(difflib.unified_diff(hbase.splitlines(True),hnew.splitlines(True),fromfile='handoff-before.md',tofile='handoff.md')))
preamble=(R/'outputs/SWINGBY_FINITE_WINDOW_CONTINUATION.tex').read_text().split('\\makeatletter')[0].replace('from-zero finite-window propagation and entry','continuation deadlines and dimension dependence')
stubs='\\makeatletter\n'+''.join('\\@namedef{r@'+k+'}{{source}{}}\n' for k in sorted(set(refs)-set(labels)))+'\\makeatother\n'
(R/'outputs/SWINGBY_BRANCH_DIMENSION_CONTINUATION.tex').write_text(preamble+stubs+'\\begin{document}\n\\maketitle\n'+body+'\n\\end{document}\n')
(R/'work/sbguard-build').mkdir(exist_ok=True)
print('Prepared staged edits, backups and standalone:',len(labels),'new labels; all new references resolve.')
