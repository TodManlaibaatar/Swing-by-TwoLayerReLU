from pathlib import Path
import hashlib, re, difflib, json

root = Path.cwd()
downloads = Path('/Users/todmanlaibaatar/Downloads')
expected = {
    'main-15.tex': '0d38f646a16da7f9e379d779714faca41445b803fdf1360cdbac8b7f7c07a5de',
    'SWINGBY_THEORY_HANDOFF_v13.md': 'bd09f6fb6f09b3ff3b8b38d0d60c872c050a75f9043abee94debcb84aa6f91c8',
}
for name, digest in expected.items():
    raw = (downloads / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, 'Incoming source changed: ' + name
    backup = root / 'outputs' / (Path(name).stem + '-before-stage0' + Path(name).suffix)
    if backup.exists():
        assert backup.read_bytes() == raw
    else:
        backup.write_bytes(raw)

base = (downloads / 'main-15.tex').read_text()
new = base
old_header = '''% SCOPE (2026-09-17): noisy many-neuron headline = specialization + predictive
% OOD normal form.  General noisy reversal signs are not load-bearing; the
% complete reversal theorem in this draft is the restricted noiseless fixed-gate case.'''
new_header = '''% STAGE 0 RE-AUDIT (2026-09-17): the original initialized noisy many-neuron
% reversal goal is reopened by the user. Scope A is not accepted as necessary.
% See sbaudit:scope: initial noisy descent is proved; the later witness and
% the common specialization/reversal parameter region remain open.'''
assert new.count(old_header) == 1
new = new.replace(old_header, new_header)
old = '''sign theorem is not claimed here; it remains a separate open dynamical
problem.  This file is a proof-progress review copy and marks all remaining
trajectory-level obligations explicitly.'''
replace = '''sign theorem is not yet proved here. A new direct argument proves initial
descent on an OOD arc with high probability in the noisy many-neuron model;
the original reversal goal is being re-audited through finite-window signed
or endpoint certificates. Whether the remaining increasing side needs more
than a short local argument is unresolved. This file is a proof-progress
review copy and marks the remaining trajectory-level obligations explicitly.'''
assert new.count(old) == 1
new = new.replace(old, replace)
anchor = r'\label{rem:scope-A-specialization}'
new = new.replace(anchor, anchor + '\n' + r'\textbf{Historical scope proposal, reopened by the Stage 0 audit in Section~\ref{sbaudit:scope}.} The following paragraph records the incoming review scope, not a proof that the original noisy reversal goal needs a separate major theory.' + '\n', 1)
anchor = r'\label{rem:T3-scope-A}'
new = new.replace(anchor, anchor + '\n' + r'\textbf{Stage 0 qualification.} Section~\ref{sbaudit:scope} reopens the original reversal goal; the incoming scope choice below is provisional. No impossibility or necessity-of-a-second-theory conclusion is asserted.' + '\n', 1)
body = (root / 'work/sbaudit-stage0.tex').read_text()
assert new.count(r'\end{document}') == 1
new = new.replace(r'\end{document}', body + '\n' + r'\end{document}')
labels = re.findall(r'\\label\{([^}]+)\}', body)
all_labels = re.findall(r'\\label\{([^}]+)\}', new)
refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', body)
assert len(set(labels)) == len(labels)
assert all(all_labels.count(x) == 1 for x in labels)
assert not set(refs) - set(all_labels), set(refs) - set(all_labels)
# Check environment nesting in the newly authored module only.
stack = []
for typ, env in re.findall(r'\\(begin|end)\{([^}]+)\}', body):
    if typ == 'begin': stack.append(env)
    else:
        assert stack and stack.pop() == env, (typ, env)
assert not stack
(root / 'outputs/main-15-stage0-staged.tex').write_text(new)
(root / 'outputs/SWINGBY_STAGE0_MAIN15.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True), new.splitlines(True), fromfile='main-15-before-stage0.tex', tofile='main-15.tex')))

hbase = (downloads / 'SWINGBY_THEORY_HANDOFF_v13.md').read_text()
hnew = (root / 'work/sbaudit-handoff.md').read_text() + '\n' + hbase
(root / 'outputs/SWINGBY_THEORY_HANDOFF_v13-stage0-staged.md').write_text(hnew)
(root / 'outputs/SWINGBY_STAGE0_HANDOFF.patch').write_text(''.join(difflib.unified_diff(hbase.splitlines(True), hnew.splitlines(True), fromfile='handoff-before-stage0.md', tofile='handoff-v13.md')))

preamble = r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools,mathrsfs}
\usepackage[colorlinks=true]{hyperref}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\title{Swing-by: Stage 0 audit and initialized noisy descent}
\author{Theory continuation for review}
\date{September 17, 2026}
'''
stubs = '\\makeatletter\n' + ''.join('\\@namedef{r@' + k + '}{{source}{}}\n' for k in sorted(set(refs) - set(labels))) + '\\makeatother\n'
(root / 'outputs/SWINGBY_STAGE0_AUDIT.tex').write_text(preamble + stubs + '\\begin{document}\n\\maketitle\n' + body + '\n\\end{document}\n')
(root / 'work/sbaudit-build').mkdir(exist_ok=True)
(root / 'outputs/SWINGBY_STAGE0_SOURCE_MANIFEST.json').write_text(json.dumps({'incoming_hashes': expected, 'new_labels': labels, 'new_references_resolve': True, 'environment_nesting': 'passed'}, indent=2) + '\n')
print(f'Staged manuscript and handoff; {len(labels)} new labels, references and environments checked.')
