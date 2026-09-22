from pathlib import Path
import hashlib,json,re,difflib
root=Path.cwd(); down=Path('/Users/todmanlaibaatar/Downloads')
expected=json.loads((root/'outputs/SWINGBY_ENDPOINT_INSTALL_MANIFEST.json').read_text())['installed_hashes']
for name,sha in expected.items():
 raw=(down/name).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==sha, 'User source changed: '+name
 backup=root/'outputs'/(Path(name).stem+'-before-coherent'+Path(name).suffix)
 if backup.exists(): assert backup.read_bytes()==raw
 else: backup.write_bytes(raw)
body='\n'.join((root/'work'/p).read_text() for p in ['sbquant-continuation.tex','sbcoherent-noisy.tex'])
base=(down/'main-15.tex').read_text(); assert base.count(r'\end{document}')==1
new=base.replace(r'\end{document}',body+'\n'+r'\end{document}')
labels=re.findall(r'\\label\{([^}]+)\}',new)
assert len(labels)==len(set(labels))
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',body)
assert not set(refs)-set(labels),set(refs)-set(labels)
stack=[]
for typ,env in re.findall(r'\\(begin|end)\{([^}]+)\}',body):
 if typ=='begin': stack.append(env)
 else: assert stack and stack.pop()==env,(typ,env)
assert not stack
assert body.count(r'\[')==body.count(r'\]')
# The entire previously installed source, including R0, is retained as a prefix.
assert new[:new.index(body)]==base[:base.index(r'\end{document}')]
(root/'outputs/main-15-coherent-staged.tex').write_text(new)
(root/'outputs/SWINGBY_COHERENT_MAIN15.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='main-15-before-coherent.tex',tofile='main-15.tex')))
supp=(root/'outputs/SWINGBY_LEARNED_MASS_ENDPOINT.tex').read_text()
assert supp.count(r'\end{document}')==1
supp=supp.replace(r'\end{document}',body+'\n'+r'\end{document}')
supp=supp.replace(r'\title{Swing-by: quantitative descent and a noisy learned-mass endpoint theorem}',r'\title{Swing-by: a noisy coherent endpoint from isotropic initialization}')
(root/'outputs/SWINGBY_COHERENT_NOISY_ENDPOINT.tex').write_text(supp)
manifest={'incoming_hashes':expected,'old_manuscript_retained_verbatim':True,'R0_supplement_section_retained':True,'labels_unique':True,'new_references_resolve':True,'environment_nesting':'passed','new_labels':re.findall(r'\\label\{([^}]+)\}',body),'full_conference_build':False,'full_conference_build_reason':'Conference style and local notation file not supplied.'}
(root/'outputs/SWINGBY_COHERENT_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Staged append-only manuscript; preserved R0; all new references resolve.')
