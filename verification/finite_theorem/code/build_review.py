from pathlib import Path
import re, hashlib, json

root=Path('/Users/todmanlaibaatar/Documents/Codex/2026-09-13/ex')
source=Path('/Users/todmanlaibaatar/Downloads/main-9.tex')
original=source.read_text()
body=(root/'work/proof-body.tex').read_text()
s=original
changes=[]

def replace_once(old,new,description):
    global s
    count=s.count(old)
    if count!=1:
        raise ValueError(f'{description}: expected one occurrence, found {count}')
    s=s.replace(old,new,1)
    changes.append(description)

replace_once(r'\usepackage{amsthm}',r'\usepackage{amsthm}'+'\n'+r'\usepackage{enumitem}',
             'Declare enumitem explicitly for the source enumerate options')
replace_once(r'\newtheorem{corollary}[theorem]{Corollary}',
             r'\newtheorem{corollary}[theorem]{Corollary}'+'\n'+r'\newtheorem{conjecture}[theorem]{Conjecture}',
             'Add an explicit environment for the unproved Theorem 1 target')

old=r'''\inf_{0<|\phi|\le\theta_1}
\frac{
-\operatorname{sgn}(\phi)V_{\mathrm{tar}}(\phi)
}{|\phi|},'''
new=r'''\inf_{|\phi|\le\theta_1}
\left(\Gamma_{\mathrm{tar}}(\phi)-T_{\mathrm{tar}}(\phi)\right),'''
replace_once(old,new,'Replace the impossible early uncentered coercivity constant')
old=r'''\inf_{0<|\phi|\le\theta_{\mathrm{post}}}
\frac{-\operatorname{sgn}(\phi)V_{\mathrm{tar}}(\phi)}{|\phi|}'''
new=r'''\inf_{|\phi|\le\theta_{\mathrm{post}}}
\left(\Gamma_{\mathrm{tar}}(\phi)-T_{\mathrm{tar}}(\phi)\right)'''
replace_once(old,new,'Replace the impossible post-trapping uncentered coercivity constant')

anchor='For the strong arc define\n'
bias=r'''For the strong arc, use the bias-aware coercivity of
Proposition~\ref{sbnew:bias} below. Define
\[
b_{\rm tar}:=
\left|\frac{\bar a_{12,\star}^{(1)}(0)+\bar a_{12,\star}^{(2)}(0)}2\right|
+\frac{\varepsilon_{12}^{(1)}+\varepsilon_{12}^{(2)}}2.
\]
This is a deterministic bound for $|V_{\rm tar}(0)|$ on the existing
gated-moment event, which is included in the event for this invocation.
The curvature infimum below may be replaced by the deterministic lower
bound in \eqref{sbnew:kappa-cert}. Define
'''
replace_once(anchor,bias,'Introduce the nonzero strong target bias and its data-event bound')
early_start=s.index(r'\begin{lemma}[Early projected trapping and weak-seed retention]')
early_end=s.index(r'\end{lemma}',early_start)
early=s[early_start:early_end]
event_old=r'''\mathcal H_{\mathrm{data}}(\delta)
\cap
\mathcal H_{\mathrm{init}}'''
assert early.count(event_old)==1
early=early.replace(event_old,r'''\mathcal H_{\mathrm{data}}(\delta)
\cap\mathcal G_{\mathrm{gate}}
\cap
\mathcal H_{\mathrm{init}}''',1)
s=s[:early_start]+early+s[early_end:]
changes.append('Include the existing gated-moment event in the corrected early handoff')
replace_once(r'''C_{1,0}
:=
a_+\sin(D_\angle)T_{1,+}''',r'''C_{1,0}
:=
a_+b_{\rm tar}
+a_+\sin(D_\angle)T_{1,+}''','Pay the bias in early strong-angle forcing')
replace_once(r'''R_{\mathrm{lock}}^{(0)}
:=
a_+^{\mathrm{post}}\sqrt{T_{1,+}^{\mathrm{post}}}\,F_\star''',r'''R_{\mathrm{lock}}^{(0)}
:=
a_+^{\mathrm{post}}b_{\rm tar}
+a_+^{\mathrm{post}}\sqrt{T_{1,+}^{\mathrm{post}}}\,F_\star''','Pay the bias in the fine-locking floor')
replace_once(r'''\Psi(D)
:={}&
a_+^{\mathrm{post}}\sin(D)T_{1,+}^{\mathrm{post}}''',r'''\Psi(D)
:={}&
a_+^{\mathrm{post}}b_{\rm tar}
+a_+^{\mathrm{post}}\sin(D)T_{1,+}^{\mathrm{post}}''','Pay the bias in the locking-gap comparison')
replace_once(r'''Q_{1,0}^{\mathrm{post}}
:={}&
a_{0,+}^{\mathrm{post}}''',r'''Q_{1,0}^{\mathrm{post}}
:={}&
a_{0,+}^{\mathrm{post}}b_{\rm tar}
+a_{0,+}^{\mathrm{post}}''','Pay the bias in post-trapping cone invariance')
replace_once('The strong target coercivity therefore gives',
             'The bias-aware target coercivity in Proposition~\\ref{sbnew:bias},\nwith its bias already included in $C_{1,0}$, therefore gives',
             'Replace the early strong-angle proof justification')
replace_once(r'''\paragraph{Step 6: strong-cone invariance and the coarse strong-angle ceiling.}
Fix $i\in A_1^{\mathrm{seed}}$.  On the bootstrap interval,''',r'''\paragraph{Step 6: strong-cone invariance and the coarse strong-angle ceiling.}
Fix $i\in A_1^{\mathrm{seed}}$. The bias-aware inequality of
Proposition~\ref{sbnew:bias} is used, with the bias included in
$Q_{1,0}^{\mathrm{post}}$. On the bootstrap interval,''',
             'Replace the post-trapping strong-angle proof justification')

start=s.index(r'\begin{proposition}[Inner strong-region cone and branch interface]')
end=s.index('% T1.10i: inner residual decomposition',start)
# Keep the following comment separator intact.
s=s[:start]+r'''\begin{remark}[Corrected inner-branch interface]
\label{prop:inner-strong-branch-interface}
The former proposition at this label incorrectly converted an additive
moving-slab error into a Lipschitz coefficient, omitted the negative-side
interval length from its contraction rate, and inferred a classical
branch across empirical jumps. Proposition~\ref{sbnew:inner} below supplies
the valid pointwise cone and pairwise spread result, with explicit jump
floor and initial transient. A smooth reference branch requires the
additional hypotheses of Proposition~\ref{sbnew:weak-comparison}; no
classical empirical branch is claimed here.
\end{remark}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''+s[end:]
changes.append('Replace the unsupported inner branch proposition with a pointer to the proved replacement')

start=s.index(r'\begin{lemma}[Inner residual-matrix decomposition]')
end=s.index(r'\end{proof}',start)+len(r'\end{proof}')
s=s[:start]+r'''\begin{lemma}[Inner residual-matrix decomposition]
\label{lem:inner-residual-decomposition}
Assume the explicit cluster-1 purity condition
\eqref{sbnew:inner-purity}. For the selected cluster-2 test gate define
\[
\mathcal R_2^{\rm gate}(\phi,t)
=\mathbb E_{2,n}[(x-f_{\parallel,t}(x))x^\top\alpha_\phi(x)].
\]
Then, including selected boundary values,
\[
\mathfrak A^\rho(\phi,t)
=\tfrac12\mathcal R_1(t)+\tfrac12\mathcal R_2^{\rm gate}(\phi,t).
\]
At a boundary the mixture residual jump is
$\pm(x-f_\parallel(x))x^\top/(2n)$, as in
Lemma~\ref{sbnew:jumps}. The entrywise constants, and the learned-output
budgets they require, are stated explicitly in
Lemma~\ref{sbnew:residual-constants}.
\end{lemma}
\begin{proof}
Split the empirical average into its two cluster averages. Purity makes
the first selector identically one. General position leaves a single
changed summand at a test-angle boundary, proving the jump formula.
\end{proof}'''+s[end:]
changes.append('Correct inner purity, residual jump, and learned-output dependencies')

old=r'''\cos\phi\,m_1^{\mathrm{data}}
-|\sin\phi|b_2
\le0
\le
\cos\phi\,(\mu_1+b_1)
+|\sin\phi|b_2.'''
new=r'''\mu_1\cos\phi-|\cos\phi|b_1-|\sin\phi|b_2
\le0\le
\mu_1\cos\phi+|\cos\phi|b_1+|\sin\phi|b_2.'''
replace_once(old,new,'Correct the partial-band box bounds on both cosine half-planes')
replace_once('Its inner boundary is the same purity edge $\\theta_C$ appearing in\n\\eqref{eq:C1}.',
             'Any strictly feasible $\\theta_C$ in \\eqref{eq:C1} is a buffered\npurity edge; it need not equal the exact partial-band boundary.',
             'Distinguish a strict purity edge from the exact partial-band boundary')

start=s.index(r'\begin{remark}[T1.10 stopping-time assembly]')
end=s.index(r'\end{remark}',start)+len(r'\end{remark}')
s=s[:start]+r'''\begin{remark}[T1.10 stopping-time assembly: still open]
\label{rem:T110-assembly}
The identities and local comparisons above do not yet exclude every
auxiliary face before the first hitting of $m_1^{\rm eff}=m_{\rm prog}$.
The proof additions below discharge global norm and perpendicular bounds,
correct the early centering and finite-sample branch interfaces, and expose
the exact remaining signed-measure and mismatch obligations. In particular,
the one-sided terminal self-field certificate alone does not imply the
absolute guard $|\mathcal X_1|\le X_C$. Theorem 1 remains an unproved
target, explicitly marked as such below.
\end{remark}'''+s[end:]
changes.append('Replace the assembly roadmap with an explicit non-closure status')

target=r'\begin{theorem}[Two-concept gate emergence and specialization]'
start=s.index(target)
end=s.index(r'\end{theorem}',start)+len(r'\end{theorem}')
th=s[start:end]
th=th.replace(target,r'\begin{conjecture}[Two-concept gate emergence and specialization: unproved target]')
th=th.replace(r'\end{theorem}',r'\end{conjecture}')
th=th.replace('there exist deterministic stopping times','there exist sample-dependent times')
s=s[:start]+th+s[end:]
changes.append('Mark the full specialization statement as an unproved target and correct the time terminology')

# Integrate a single copy of the completed proof additions before the target.
insert_at=s.index(r'\begin{conjecture}[Two-concept gate emergence')
integrated=body.replace(r'\section{',r'\subsection{')
s=s[:insert_at]+integrated+'\n\n'+s[insert_at:]
s,proof_heading_fixes=re.subn(r'(\\begin\{proof\})(\s*)(?=\\paragraph)',r'\1\\leavevmode\2',s)
if proof_heading_fixes:
    changes.append(f'Add leavevmode before {proof_heading_fixes} inherited proof-opening paragraph headings')
double_superscript=r'\widetilde u_i^\angle^\top'
double_count=s.count(double_superscript)
s=s.replace(double_superscript,r'(\widetilde u_i^\angle)^\top')
if double_count:
    changes.append(f'Parenthesize {double_count} inherited double-superscript tangent transposes')
s='% REVIEW COPY: proof progress, not a completed proof of Theorem 1.\n'+s
(root/'outputs/main-10-review.tex').write_text(s)

preamble=r'''\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb,mathtools,amsthm,enumitem}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small Swing-by ReLU: proof progress}
\fancyhead[R]{\small Theorem 1 remains open}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}
\setlength{\emergencystretch}{2em}
\newtheorem{lemma}{Lemma}[section]
\newtheorem{proposition}[lemma]{Proposition}
\newtheorem{corollary}[lemma]{Corollary}
\theoremstyle{remark}\newtheorem{remark}[lemma]{Remark}
\allowdisplaybreaks[2]
\title{Swing-by ReLU\\Proof corrections and completed auxiliary results}
\author{Working supplement to the supplied \texttt{main-9.tex}}
\date{September 13, 2026}
\begin{document}
\maketitle
'''
(root/'outputs/SWINGBY_PROOF_PROGRESS.tex').write_text(preamble+body+'\n\\end{document}\n')
check_preamble=preamble.split(r'\begin{document}')[0]
check_preamble=check_preamble.replace('amsthm,enumitem','amsthm,enumitem,mathrsfs')
check_preamble=check_preamble.replace(r'\newtheorem{lemma}{Lemma}[section]',
    r'\newtheorem{theorem}{Theorem}[section]'+'\n'+r'\newtheorem{lemma}[theorem]{Lemma}')
for env in ['proposition','corollary','remark']:
    check_preamble=check_preamble.replace('\\newtheorem{'+env+'}[lemma]',
                                         '\\newtheorem{'+env+'}[theorem]')
for env,title in [('definition','Definition'),('assumption','Assumption'),('conjecture','Conjecture')]:
    check_preamble+='\n\\newtheorem{'+env+'}[theorem]{'+title+'}\n'
(root/'work/appendix-check.tex').write_text(check_preamble+'\\begin{document}\n'+s[s.index(r'\section{Notation}'):])
(root/'work/review_changes.json').write_text(json.dumps({
 'source_sha256':hashlib.sha256(original.encode()).hexdigest(),
 'changes':changes,
 'source_lines':len(original.splitlines()),
 'review_lines':len(s.splitlines())},indent=2))

# Source checks are structural, not a substitute for mathematical review.
for name,text in [('supplement',preamble+body),('review',s)]:
    labels=re.findall(r'\\label\{([^}]+)\}',text)
    duplicates=sorted({v for v in labels if labels.count(v)>1})
    refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text)
    unresolved=sorted(set(refs)-set(labels))
    print(name, 'duplicate labels:',duplicates,'unresolved refs:',unresolved)
    assert not duplicates
    if name=='supplement': assert not unresolved
print('Created integrated review and standalone proof supplement.')
