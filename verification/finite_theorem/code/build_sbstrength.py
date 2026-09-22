from pathlib import Path
import hashlib,json,re,difflib
root=Path.cwd();down=Path('/Users/todmanlaibaatar/Downloads');out=root/'outputs'
expected=json.loads((out/'SWINGBY_COHERENT_INSTALL_MANIFEST.json').read_text())['installed_hashes']
for name,sha in expected.items():
 raw=(down/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha,'User source changed: '+name
 backup=out/(Path(name).stem+'-before-strengthening'+Path(name).suffix)
 if backup.exists():assert backup.read_bytes()==raw
 else:backup.write_bytes(raw)
body=(root/'work/sbstrengthening.tex').read_text();base=(down/'main-15.tex').read_text()
old='''A new restricted noisy many-neuron theorem proves high-probability initial
descent, an order-one learned strong cohort, and a subsequent strict endpoint
rebound on an explicit probe sector in an extreme concept-separation region.
It does not establish a learned weak family or reversal at a fixed concept
ratio. The original two-family reversal goal remains open.'''
new='''A restricted noisy many-neuron theorem now certifies, with probability at
least $0.97$, an initial OOD transient, a dominant cohort with learned mass
at least $0.55$, and later strict degradation on a deterministic sector of
width $3.65\\cdot10^{-4}$ radians. An explicit regime uses $h\\ge10^7$,
$h\\varepsilon^2=0.04$, concept ratio $10^{-4}$, and noise $10^{-9}$.
On the certified time interval the error minimum precedes the declared
specialization time; specialization is not proved to cause the reversal.
The weak concept remains unlearned. R0 and the earlier coherent theorem
are retained, and the original two-learned-family goal remains open.'''
assert old in base
changed=base.replace(old,new).replace(r'\end{document}',body+'\n'+r'\end{document}')
labels=re.findall(r'\\label\{([^}]+)\}',changed);assert len(labels)==len(set(labels))
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',body);assert not set(refs)-set(labels)
stack=[]
for typ,env in re.findall(r'\\(begin|end)\{([^}]+)\}',body):
 if typ=='begin':stack.append(env)
 else:assert stack.pop()==env
assert not stack
assert body.count(r'\[')==body.count(r'\]')
assert changed.replace(body+'\n','').replace(new,old)==base
(out/'main-15-strengthened-staged.tex').write_text(changed)
(out/'SWINGBY_STRENGTHENING_MAIN15.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),changed.splitlines(True),fromfile='main-15-before-strengthening.tex',tofile='main-15.tex')))
supp=(out/'SWINGBY_COHERENT_NOISY_ENDPOINT.tex').read_text()
supp=supp.replace(r'\end{document}',body+'\n'+r'\end{document}').replace(r'\date{September 17, 2026}',r'\date{September 18, 2026}')
supp=supp.replace(r'\title{Swing-by: a noisy coherent endpoint from isotropic initialization}',r'\title{Swing-by: sharper certificates and a smaller-width noisy theorem}')
(out/'SWINGBY_STRENGTHENED_NOISY_ENDPOINT.tex').write_text(supp)
manifest={'incoming_hashes':expected,'R0_retained_verbatim':True,'Theorem_5_4_retained_verbatim':True,'changes':'New Section 6 plus precise abstract scope; prior proof body unchanged.','labels_unique':True,'references_resolve':True,'environment_nesting':'passed','full_conference_build':False,'reason':'Missing conference style and notation files.'}
(out/'SWINGBY_STRENGTHENING_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Staged manuscript and standalone supplement, with preserved R0 and Theorem 5.4.')
