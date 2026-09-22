from pathlib import Path
import hashlib,re,difflib,json
root=Path.cwd(); downloads=Path('/Users/todmanlaibaatar/Downloads')
expected=json.loads((root/'outputs/SWINGBY_STAGE0_INSTALL_MANIFEST.json').read_text())['installed_hashes']
for name,digest in expected.items():
 raw=(downloads/name).read_bytes(); assert hashlib.sha256(raw).hexdigest()==digest,'Source changed: '+name
 backup=root/'outputs'/(Path(name).stem+'-before-endpoint'+Path(name).suffix)
 if backup.exists(): assert backup.read_bytes()==raw
 else: backup.write_bytes(raw)
body='\n'.join((root/'work'/n).read_text() for n in ['sbend-continuation.tex','sbsep-continuation.tex'])
base=(downloads/'main-15.tex').read_text()
new=base.replace(r'r_\xi,mathsf K',r'r_\xi,\mathsf K').replace('''% See sbaudit:scope: initial noisy descent is proved; the later witness and
% the common specialization/reversal parameter region remain open.''','''% See sbsep:theorem: a noisy many-neuron learned-strong-cohort endpoint theorem
% is proved in an extreme-separation region. Two learned families and a fixed
% concept-ratio regime remain open; no headline scope pivot has been accepted.''')
old='''sign theorem is not yet proved here. A new direct argument proves initial
descent on an OOD arc with high probability in the noisy many-neuron model;
the original reversal goal is being re-audited through finite-window signed
or endpoint certificates. Whether the remaining increasing side needs more
than a short local argument is unresolved.'''
rep='''sign theorem in the original two-family regime is not yet proved here.
A new restricted noisy many-neuron theorem proves high-probability initial
descent, an order-one learned strong cohort, and a subsequent strict endpoint
rebound on an explicit probe sector in an extreme concept-separation region.
It does not establish a learned weak family or reversal at a fixed concept
ratio. The original two-family reversal goal remains open.'''
assert new.count(old)==1;new=new.replace(old,rep)
for label in ['sbaudit:scope','sbaudit:later-witness']:
 anchor='\\label{'+label+'}'
 assert new.count(anchor)==1
 new=new.replace(anchor,anchor+'\n'+r'\textbf{Continuation update.} Section~\ref{sbsep:scope} now proves a restricted noisy learned-strong-cohort endpoint theorem. The following Stage 0 status is historical; the original two-family goal remains open.'+'\n')
assert new.count(r'\end{document}')==1
new=new.replace(r'\end{document}',body+'\n'+r'\end{document}')
labels=re.findall(r'\\label\{([^}]+)\}',body)
all_labels=re.findall(r'\\label\{([^}]+)\}',new)
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',body)
assert len(labels)==len(set(labels))
assert all(all_labels.count(x)==1 for x in labels)
assert not set(refs)-set(all_labels),set(refs)-set(all_labels)
stack=[]
for typ,env in re.findall(r'\\(begin|end)\{([^}]+)\}',body):
 if typ=='begin':stack.append(env)
 else:assert stack and stack.pop()==env,(typ,env)
assert not stack
# Check display delimiters separately; they are not environments.
assert body.count(r'\[')==body.count(r'\]')
(root/'outputs/main-15-endpoint-staged.tex').write_text(new)
(root/'outputs/SWINGBY_ENDPOINT_MAIN15.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='main-15-before-endpoint.tex',tofile='main-15.tex')))
hbase=(downloads/'SWINGBY_THEORY_HANDOFF_v13.md').read_text()
hnew=(root/'work/sbend-handoff.md').read_text()+hbase
(root/'outputs/SWINGBY_THEORY_HANDOFF_v13-endpoint-staged.md').write_text(hnew)
(root/'outputs/SWINGBY_ENDPOINT_HANDOFF.patch').write_text(''.join(difflib.unified_diff(hbase.splitlines(True),hnew.splitlines(True),fromfile='handoff-before-endpoint.md',tofile='handoff-v13.md')))
# Include the earlier Stage 0 results so the supplement is independently reviewable.
supp=(root/'work/sbaudit-stage0.tex').read_text()+'\n'+body
supp=supp.replace(r'\label{sbaudit:scope}',r'\label{sbaudit:scope}'+'\n'+r'\textbf{Historical audit.} The restricted learned-mass result in Section~\ref{sbsep:scope} below supersedes the missing-later-witness status in this earlier section.')
sl=set(re.findall(r'\\label\{([^}]+)\}',supp));sr=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',supp))
stubs='\\makeatletter\n'+''.join('\\@namedef{r@'+k+'}{{source}{}{}{source-anchor}{}}\n' for k in sorted(sr-sl))+'\\makeatother\n'
pre=r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools,mathrsfs}
\usepackage[colorlinks=true]{hyperref}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\title{Swing-by: quantitative descent and a noisy learned-mass endpoint theorem}
\author{Theory continuation for review}
\date{September 17, 2026}
'''
(root/'outputs/SWINGBY_LEARNED_MASS_ENDPOINT.tex').write_text(pre+stubs+'\\begin{document}\n\\maketitle\n'+r'\hypertarget{source-anchor}{Cross-references marked ``source'' refer to the accompanying manuscript.}'+'\n'+supp+'\n\\end{document}\n')
(root/'work/sbend-build').mkdir(exist_ok=True)
manifest={'incoming_hashes':expected,'new_labels':labels,'references_resolve':True,'environment_nesting':'passed','display_delimiters':'passed','full_conference_build':False,'reason':'conference style and local notation file unavailable'}
(root/'outputs/SWINGBY_ENDPOINT_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Staged manuscript and handoff;',len(labels),'new labels; references and environment checks passed.')
