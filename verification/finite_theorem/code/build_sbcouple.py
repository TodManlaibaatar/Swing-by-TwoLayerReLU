from pathlib import Path
import hashlib, difflib, re
R=Path.cwd(); D=Path('/Users/todmanlaibaatar/Downloads')
expected={'main-14.tex':'ed1322a42e21c286671e2e79c9c5df25f5c065d79372eb028ec3a4aa7d4208c7','SWINGBY_THEORY_HANDOFF_v12.md':'97770d08e407a97c8345dd9a1c61f2487815660f3feafb8f45499383d5c2ac7e'}
for name,h in expected.items():
 p=D/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,'Source changed; reconcile first'
 backup=R/'outputs'/(p.stem+'-before-coupled-signed-source'+p.suffix)
 if backup.exists(): assert backup.read_bytes()==p.read_bytes()
 else: backup.write_bytes(p.read_bytes())
body=(R/'work/sbcouple-continuation-body.tex').read_text()
base=(D/'main-14.tex').read_text()
assert r'\label{sbcouple:scope}' not in base
new=base.replace(r'\end{document}',body+'\n'+r'\end{document}')
# Mark the former first-remaining-estimate explicitly as historical.
old=r'\begin{remark}[The first remaining estimate, after consuming the identities]'
assert new.count(old)==1
new=new.replace(old,r'\begin{remark}[Transport frontier before the coupled signed-source continuation]')
anchor=r'\label{sbflux:frontier}'
new=new.replace(anchor,anchor+'\n'+r'The source decomposition and coupled propagation requested here are now supplied in Section~\ref{sbcouple:scope}. The remaining initialized audit is \eqref{sbcouple:unresolved-budget}; the following records why the preceding transport identity alone did not close it.'+'\n')
newlabels=re.findall(r'\\label\{([^}]+)\}',body)
assert len(newlabels)==len(set(newlabels))
alllabels=re.findall(r'\\label\{([^}]+)\}',new)
assert all(alllabels.count(x)==1 for x in newlabels)
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',body)
assert not set(refs)-set(alllabels),set(refs)-set(alllabels)
(R/'outputs/main-14-coupled-signed-source-staged.tex').write_text(new)
(R/'outputs/SWINGBY_COUPLED_SIGNED_MAIN14.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='main-14-before.tex',tofile='main-14.tex')))
handoff=(D/'SWINGBY_THEORY_HANDOFF_v12.md').read_text()
anchor='## September 16 transport continuation — updated proof frontier'
assert handoff.count(anchor)==1
handoff=handoff.replace(anchor,(R/'work/sbcouple-handoff-update.md').read_text()+'\n## Earlier September 16 transport continuation — frontier superseded above')
(R/'outputs/SWINGBY_THEORY_HANDOFF_v12-coupled-signed-source-staged.md').write_text(handoff)
preamble=(R/'outputs/SWINGBY_WEIGHTED_TRANSPORT_CONTINUATION.tex').read_text().split(r'\begin{document}')[0]
preamble=preamble.replace(r'\usepackage{amsmath,amssymb,amsthm,mathtools}',r'\usepackage{amsmath,amssymb,amsthm,mathtools,mathrsfs}')
preamble=preamble.replace('fixed-complement transport and concurrent output bounds','signed cohort sources and coupled complement propagation')
# Standalone external-reference placeholders are identified explicitly.
external=sorted(set(refs)-set(newlabels))
stubs='\\makeatletter\n'+''.join('\\@namedef{r@'+x+'}{{source}{}}\n' for x in external)+'\\makeatother\n'
(R/'outputs/SWINGBY_COUPLED_SIGNED_CONTINUATION.tex').write_text(preamble+stubs+'\\begin{document}\n\\maketitle\n'+body+'\n\\end{document}\n')
(R/'work/sbcouple-build').mkdir(exist_ok=True)
print('Prepared staged manuscript, handoff, backups and standalone proof. New labels:',len(newlabels),'All new references resolve in manuscript.')
