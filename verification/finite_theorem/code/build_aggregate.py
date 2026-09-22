from pathlib import Path
import re

root = Path('/Users/todmanlaibaatar/Documents/Codex/2026-09-13/ex')
attachments = Path('/Users/todmanlaibaatar/.codex/attachments')
c = (attachments/'73df94f0-49fd-4807-a2fa-fe2e38edb729/pasted-text.txt').read_text()
d = (attachments/'620cbcf4-3d9c-4e34-afbd-c3dfa373e7ba/pasted-text.txt').read_text()

def lemma_and_proof(text, title):
    start = text.index('\\begin{lemma}[' + title)
    lemma_end = text.index('\\end{lemma}', start) + len('\\end{lemma}')
    proof_start = text.index('\\begin{proof}', lemma_end)
    proof_end = text.index('\\end{proof}', proof_start) + len('\\end{proof}')
    return text[start:lemma_end] + '\n\n' + text[proof_start:proof_end]

intro = r'''
\section{Aggregate propagation toward the direct Theorem 3 route}
\label{sbagg:scope}
This addition proves deterministic aggregate estimates for the true empirical
gradient flow. It does not establish that their accuracy budget holds through
the learning and reversal interval. All notation in this section is local.
There are $n$ samples in each cluster, all in $\mathcal S$, and
$\mathbb E_n=(\mathbb E_{1,n}+\mathbb E_{2,n})/2$.
Set $\Sigma_n=\mathbb E_n[xx^\top]$, $\lambda_n=\|\Sigma_n\|_{\rm op}$,
and $\|g\|_n^2=\mathbb E_n\|g(x)\|^2$. Use the unscaled network
$f(x)=\sum_iw_i(u_i^\top x)_+$ and loss
$\mathcal L=\frac12\|x-f\|_n^2$.
Write $v_i=P_{\mathcal S}u_i$, $z_i=P_{\mathcal S}w_i$,
$e=x-f_\parallel$, and $\delta_i=v_i-z_i$.
Compatible selected gates $\alpha_i\in[0,1]$ give
\[
 A_i^\rho=\mathbb E_n[e x^\top\alpha_i],\qquad
 P_{i,\perp}^\rho=\mathbb E_n[x\alpha_i
                  \langle w_{i,\perp},f_\perp(x)\rangle],
 \qquad \dot z_i=A_i^\rho v_i,\quad
 \dot v_i=(A_i^\rho)^\top z_i-P_{i,\perp}^\rho.
\]
For a set $J$, let $M_J=\sum_J z_iv_i^\top$, $B_J=\sum_Jz_iz_i^\top$,
and $C_J=\sum_Jv_iv_i^\top$.
All differential inequalities below hold almost everywhere along the
absolutely continuous trajectory and imply their integrated comparisons.
At zero norms these follow by regularization. The notation $D^+$ is used
only in this almost-everywhere comparison sense, without claiming a
continuous empirical angular field or a derivative at every gate switch.
'''

jensen = lemma_and_proof(c, 'Cross-residual energy controls weighted residual forcing')
coupled = lemma_and_proof(c, 'Coupled propagation of cross residual and aggregate axis grading')
optimal = lemma_and_proof(d, 'Optimal two-axis partition and aggregate checkpoint')
pure = lemma_and_proof(d, 'Pure positive-sector cluster-$2$ residual reduction')
pure = pure.replace('Pure positive-sector cluster-$2$ residual reduction',
                    'Quadrant-II cluster-$2$ pure-sector residual reduction')
pure = pure.replace('u^\\top\\mathcal R_2u^\\angle',
                    '\\widetilde u(\\phi)^\\top\\mathcal R_2\\widetilde u^\\angle(\\phi)')
pure = pure.replace('For $\\phi\\in J_{C,2}^{\\mathrm{pure}}$,',
                    'Define the aligned angular proxy by '
                    '$V_{C,2}^\\rho=\\widetilde u^\\top(\\mathcal R_2/2)'
                    '\\widetilde u^\\angle$, where '
                    '$\\widetilde u^\\angle=(-\\sin\\phi,\\cos\\phi)$. '
                    'It is not the actual mismatched neuron angle velocity. '
                    'For $\\phi\\in J_{C,2}^{\\mathrm{pure}}$,')

extras = r'''
\subsection{A sharper comparison and the required accuracy condition}
\begin{remark}[Available global ceilings do not require extra trajectory hypotheses]
\label{sbagg:global-ceilings}
Under the source's balanced initialization and perpendicular contraction,
the ceilings in the coupled lemma have explicit choices. Put
$X_{\max}=\max_{\mathcal D}\|x\|$ and $s_0=h\varepsilon^2$.
On $[t_0,t_0+T]$ one may use
\[
 S_*=s_0+\tfrac12C_x(t_0+T),\qquad
 R_{n,*}=\sqrt{C_x}(1+s_0),\qquad
 R_{\infty,*}=(1+S_*)X_{\max},\qquad
 \Pi_*=\lambda_n h\varepsilon^2\sqrt{S_*}.
\]
Indeed balance and the exact mass identity give
$\dot S=C_x/2-2\|f-x/2\|_n^2\le C_x/2$.
The pointwise estimate $\|f(x)\|\le S\|x\|$ and full loss dissipation
give the residual ceilings. Finally
$\|W_\perp\|_F\le\varepsilon\sqrt h$ implies
$\|f_\perp\|_n\le\varepsilon\sqrt{h\lambda_n S_*}$, and
Cauchy--Schwarz gives
$\Pi\le\sqrt{\lambda_n}\|f_\perp\|_n\|W_\perp\|_F$.
These choices establish finite ceilings; their quantitative usefulness
in the accuracy condition below is a separate question.
\end{remark}

\begin{remark}[The smaller input-kernel coefficient is also valid]
\label{sbagg:kernel-sharpening}
The preceding proof safely uses $\Lambda_u=\lambda_1+\lambda_2$.
In fact all its conclusions remain true with
$\Lambda_u=\max\{\lambda_1,\lambda_2\}$.
To see this, the correct-to-wrong input-kernel block of one neuron,
with the common cluster normalization understood, is
\[
 \frac12\begin{pmatrix}
 z_{i1}z_{i2}T_{i,1}^*T_{i,1}&z_{i2}^2T_{i,1}^*T_{i,2}\\
 z_{i1}^2T_{i,2}^*T_{i,1}&z_{i1}z_{i2}T_{i,2}^*T_{i,2}
 \end{pmatrix}.
\]
The block-diagonal part has norm at most
$|z_{i1}z_{i2}|\max_p\lambda_p/2$.
The off-diagonal part has norm at most
$\tau_u\max\{z_{i1}^2,z_{i2}^2\}/2
\le\tau_u\|z_i\|^2/2$.
Add these bounds and sum over neurons. Thus the proposed replacement
of the maximum by the sum was sufficient but not necessary.
\end{remark}

\begin{corollary}[Two-dimensional comparison and finite accuracy budget]
\label{sbagg:budget}
Use the constants of Lemma~\ref{lem:coupled-cross-transverse}, and set
\[
 Y=Y_1+Y_2,\quad c=\sqrt{S_*}U_*,\quad
 b=\sqrt{S_*}R_{\infty,*}\nu_\times,\quad
 H=\begin{pmatrix}0&a_Y\\2c&K_*\end{pmatrix}.
\]
Then, componentwise,
\[
 \binom{X(t)}{Y(t)}\le
 e^{H(t-t_0)}\binom{X(t_0)}{Y(t_0)}+
 \int_{t_0}^t e^{H(t-s)}
 \binom{a_{\rm data}+\sqrt{S_*C_x}\Pi(s)}
 {2b+\Pi_1(s)+\Pi_2(s)}\,ds.
\]
In particular a sufficient, fully quantitative condition for
$X+Y\le C_{\rm tar}\eta$ on $[t_0,t_0+T]$ is
\begin{equation}\label{sbagg:accuracy-budget}
 e^{L_{\rm cpl}T}Z(t_0)+F_*\mathcal E_{L_{\rm cpl}}(T)
 \le C_{\rm tar}\eta,
 \qquad
 \mathcal E_a(T)=\begin{cases}(e^{aT}-1)/a,&a>0,\\T,&a=0.\end{cases}
\end{equation}
The matrix comparison can replace the scalar left side by a smaller bound.
If $\eta$ varies, an $O(\eta)$ claim requires a constant $C_{\rm tar}$
uniform in that variation. Finite constants for each fixed interval do
not establish such uniformity when $T$ depends on $\eta$ or $\varepsilon$.
\end{corollary}
\begin{proof}
Add the two $Y_p$ inequalities and retain the separate $X$ inequality.
The off-diagonal entries of $H$ are nonnegative, so its exponential
preserves componentwise order; variation of constants proves the first
claim. The scalar comparison proves \eqref{sbagg:accuracy-budget}.
The largest eigenvalue of $H$ is
$(K_*+\sqrt{K_*^2+8a_Yc})/2$; this improvement still permits growth.
\end{proof}

\begin{corollary}[Consequent grading and coefficient rates]
\label{sbagg:rates}
Fix $p$ and use the ordered basis $(e_p,e_{3-p})$. Suppose on the same
interval $Y_p\le\eta C$ and $\mathcal F_{J_p}^{(p)}+\Pi_p\le\eta f$,
with $\eta>0$ fixed in time. Then write
\[
 M_{J_p}=\begin{pmatrix}a&\eta b_0\\\eta c_0&\eta^2d\end{pmatrix}.
\]
The bounds below hold uniformly, with derivatives almost everywhere:
\[
 |a|\le S_*,\quad |b_0|,|c_0|\le\sqrt{S_*}C,\quad |d|\le C^2/2,
\]
\[
 |\dot a|\le2K_*S_*+\sqrt{S_*}\eta f,\qquad
 |\dot d|\le K_*C^2+Cf,
\]
\[
 |\dot b_0|,|\dot c_0|
 \le K_*\sqrt{2S_*}C+K_*\eta C^2+\sqrt{S_*}f.
\]
Also $(B_{J_p})_{22},(C_{J_p})_{22}\le\eta^2C^2$ in this ordered basis.
The forcing constant can be chosen as
\[
 f=\sqrt{S_*}\left(U_*C_{\rm tar}
              +R_{\infty,*}\frac{\nu_\times}{\eta}\right)
       +\frac{\Pi_*}{\eta}
\]
whenever \eqref{sbagg:accuracy-budget} holds.
\end{corollary}
\begin{proof}
Cauchy--Schwarz gives the static bounds. The exact identity
\[
 \dot M_J=\sum_{i\in J}
 \left(A_i^\rho v_iv_i^\top+z_iz_i^\top A_i^\rho
                         -z_i(P_{i,\perp}^\rho)^\top\right)
\]
gives
$|\dot M_{22}|\le K_*Y_p^2+Y_p(\mathcal F_{J_p}^{(p)}+\Pi_p)$.
For each off-diagonal entry the terms are bounded by
$K_*\sqrt{2S_*}Y_p+K_*Y_p^2+sqrt{S_*}(\mathcal F_{J_p}^{(p)}+\Pi_p)$.
Finally $|\dot M_{11}|\le2K_*S_*+\sqrt{S_*}\Pi_p$.
Divide by the indicated fixed powers of $\eta$ and apply the Jensen lemma.
These are bounds for the partition matrices, not automatically for the
matrix of an arbitrary probe activation cell.
\end{proof}

\subsection{Checkpoint construction and its precise limitation}
'''

checkpoint = r'''
\begin{corollary}[Cohort and primitive checkpoint certificates]
\label{sbagg:primitive}
At a fixed checkpoint let $[h]=B_1\sqcup B_2\sqcup R$ and define
\[
 Q_1=\sum_{B_1}v_{i2}^2,\quad Q_2=\sum_{B_2}v_{i1}^2,\quad
 \Delta_p=\sum_{B_p}\|v_i-z_i\|^2,\quad
 S_R=\sum_R(\|v_i\|^2+\|z_i\|^2).
\]
Then
\[
 \mathfrak D_{\rm ax}\le
 2Q_1+\Delta_1+2\sqrt{Q_1\Delta_1}
 +2Q_2+\Delta_2+2\sqrt{Q_2\Delta_2}+S_R/2.
\]
Independently, balance gives
$\mathfrak D_{\rm ax}(t)\le S(t)=\sum_i\|u_i(t)\|^2
=\sum_i\|w_i(t)\|^2$.
Suppose valid early bounds on $[0,H_\star]$ are
\[
 S(t)\le \frac{h\varepsilon^2e^{2\lambda H_\star}}
                         {D_\star^{\rm Ric}}=:S_\sharp^*,
 \qquad \sup_{x\in\mathcal D}\|f_t(x)\|\le F_\star,
 \qquad D_\star^{\rm Ric}>0.
\]
For any $t_\sharp\le H_\star$, the additional conditions
\[
 S_\sharp^*\le c_{\rm ax}^2\eta^2,\qquad
 \nu_\times+F_\star\le c_\times\eta
\]
imply, for the minimizing partition frozen at $t_\sharp$,
\[
 Z(t_\sharp)\le(c_\times+\sqrt2c_{\rm ax})\eta.
\]
This certifies a checkpoint only. It neither proves
\eqref{sbagg:accuracy-budget} on a later learning interval nor supplies
probe gate orientation, longitudinal mass, or a reversal sign.
\end{corollary}
\begin{proof}
On $B_p$ the output transverse norm is at most
$\sqrt{Q_p}+\sqrt{\Delta_p}$; square and add the input transverse energy.
On $R$, the smaller cost is at most half the total squared norm.
Applying this last observation to every neuron gives
$\mathfrak D_{\rm ax}\le S$.
The triangle inequality gives
$X=\|P_\times(x-f_\parallel)\|_n\le\nu_\times+F_\star$.
Apply Lemma~\ref{lem:optimal-axis-checkpoint}.
\end{proof}

\begin{remark}[Why small initial mass does not close late propagation]
\label{sbagg:small-mass-example}
Consider the deterministic planar data $x^{(1)}=e_1,x^{(2)}=e_2$ with
one balanced neuron $u=w=\sqrt{S/2}(1,1)^\top$. This is an algebraic
example within the balanced equations, not a counterexample to a
high-probability assertion under isotropic random initialization.
Both training gates are on, and the exact gradient flow gives
\[
 \dot S=S(1-S),\qquad
 S(t)=\frac{S(0)e^t}{1-S(0)+S(0)e^t}.
\]
Here $\nu_\times=\tau_w=\tau_u=\Pi=0$, so $F_*=0$, while
\[
 \mathfrak D_{\rm ax}=S,\qquad Y_1+Y_2=\sqrt S,\qquad X=S/2.
\]
Taking $S(0)=\eta^2<1/2$ gives the required small checkpoint
$Z(0)=\eta+\eta^2/2$. At
$t_\eta=\log((1-\eta^2)/\eta^2)$, however,
\[
 S(t_\eta)=1/2,\qquad Z(t_\eta)=1/\sqrt2+1/4.
\]
Thus a checkpoint and an $O(\eta)$ forcing budget alone cannot give
uniform $O(\eta)$ grading on a growing learning horizon.
The proved comparison correctly allows its constant to grow.
To derive useful late-time grading one must certify the finite accuracy
budget, or retain more favorable signed or gate-dependent growth terms.
This calculation does not show that such a sharper route is impossible.
\end{remark}

\subsection{The weak pure-sector calculation}
Put $\widetilde u(\phi)=(\cos\phi,\sin\phi)$. The following is conditional
on the displayed coordinate envelopes; it asserts no retention on its own.
Here $L_{\delta_{\mathrm{strong}}}>0$ denotes the supplied envelope factor.
The coordinate-envelope event means that every cluster-1 sample satisfies
$x_1\ge m_1^{\rm data}$ and $|x_2|\le b_2$, and every cluster-2 sample
satisfies $|x_1|\le b_1$ and $x_2\ge m_2^{\rm data}$, with the constants
defined in the next lemma.
'''

weak = r'''
\begin{corollary}[Weak pure-sector absolute transverse energy]
\label{sbagg:weak-Q}
Let a fixed nonempty set $B$ remain in $J_{C,2}^{\rm pure}$ on $[a,b]$.
Write $R_{rs}=R_{rs}^{(2)}$ and define
\[
 M=\sum_{B}v_{i2}^2,\quad Q=\sum_Bv_{i1}^2,\quad
 \Delta=\sum_B\|\delta_i\|^2,\quad
 \mathcal P=\left(\sum_B\|P_{i,\perp}^\rho\|^2\right)^{1/2}.
\]
The exact coordinate equations are
\[
 \dot v_{i1}=\tfrac12[R_{11}(v_{i1}-\delta_{i1})
                       +R_{21}(v_{i2}-\delta_{i2})]-P_{i1},
\]
\[
 \dot v_{i2}=\tfrac12[R_{12}(v_{i1}-\delta_{i1})
                       +R_{22}(v_{i2}-\delta_{i2})]-P_{i2}.
\]
Consequently, in the almost-everywhere comparison sense,
\[
 \frac{d}{dt}\sqrt M\ge
 \tfrac12R_{22}\sqrt M-\tfrac12|R_{12}|\sqrt Q
 -\tfrac12\sqrt{R_{12}^2+R_{22}^2}\sqrt\Delta-\mathcal P,
\]
\[
 \frac{d}{dt}\sqrt Q\le
 \tfrac12|R_{21}|\sqrt M+\tfrac12|R_{11}|\sqrt Q
 +\tfrac12\sqrt{R_{11}^2+R_{21}^2}\sqrt\Delta+\mathcal P.
\]
If $|R_{11}|\le\beta_{11}$, $|R_{21}|\le\beta_{21}$,
$M\le S_*$, $\sqrt\Delta\le D_B$, and $\mathcal P\le P_B$, then
\[
 \sqrt{Q(t)}\le e^{\beta_{11}(t-a)/2}\sqrt{Q(a)}
       +H_B\mathcal E_{\beta_{11}/2}(t-a),
\]
\[
 H_B=\tfrac12\beta_{21}\sqrt{S_*}
      +\tfrac12\sqrt{\beta_{11}^2+\beta_{21}^2}D_B+P_B.
\]
If $\sup_{\mathcal D}\|f_t\|\le F_\star$, valid choices are
\[
 \beta_{11}=s_{11}^{(2)}+F_\star\sqrt{s_{11}^{(2)}},\qquad
 \beta_{21}=|s_{12}^{(2)}|+F_\star\sqrt{s_{11}^{(2)}}.
\]
In the limit $R_{11}=R_{12}=R_{21}=0$, $\delta_i=P_i=0$,
one has $\dot Q=0$ and $\dot M=R_{22}M$.
\end{corollary}
\begin{proof}
Substitute $A_i^\rho=\mathcal R_2/2$ in the input equation. Take the
Euclidean norm over $B$ in each coordinate and use Cauchy--Schwarz for
the mismatch and perpendicular terms. Integrate the resulting scalar
inequality for $\sqrt Q$. Finally
$R_{r1}=s_{r1}^{(2)}-\mathbb E_{2,n}[f_r x_1]$ and
$|\mathbb E_{2,n}[f_r x_1]|\le F_\star\sqrt{s_{11}^{(2)}}$.
\end{proof}

\begin{remark}[Retention and the interface to Theorem 3 remain separate]
The pure arc is antipodal to the cluster-1-only arc under the same envelope
conditions. If a valid invocation of the bias-corrected T1.8--T1.9 proves
retention and its arcs satisfy
\[
 J_W^{\rm seed}\Subset J_W^{\rm ret}\Subset J_{C,2}^{\rm pure},
\]
then the pure-sector identities apply on that invocation's horizon.
But changing the arcs changes their buffers, angular suprema, and retention
budgets. Their compatibility, seed coverage, and a nonempty admissible
parameter region must be checked; no unconditional handoff is proved here.

The optimal partition is not a partition by actual probe gates. For a fixed
probe cell, $M_\omega=\sum_i\omega_i z_iv_i^\top$ can contain longitudinal
mass from both axis groups. Axis defect alone neither controls its required
transverse diagonal nor the signs of the coefficient evolution. One must
establish the cell's active-group contributions and then prove the
$\mathcal G$ sign margins and derivative remainder needed for reversal.
No identification of $\eta$ with the noise scale is used above.
\end{remark}
'''

body = '\n\n'.join([intro,jensen,coupled,extras,optimal,checkpoint,pure,weak])
body = body.replace('\\begin{proof}\n\n\\paragraph',
                    '\\begin{proof}\\leavevmode\n\n\\paragraph')
labels = re.findall(r'\\label\{([^}]+)\}',body)
for label in labels:
    if not label.startswith('sbagg:'):
        body = body.replace('{'+label+'}', '{sbagg:'+label+'}')

preamble = r'''\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb,mathtools,amsthm,enumitem}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\newtheorem{lemma}{Lemma}[section]
\newtheorem{corollary}[lemma]{Corollary}
\theoremstyle{remark}\newtheorem{remark}[lemma]{Remark}
\allowdisplaybreaks[2]
\title{Swing-by: aggregate propagation and checkpoint audit}
\author{Deterministic proof addition to the main-9 review}
\date{September 14, 2026}
\begin{document}
\maketitle
'''
(root/'work/aggregate-body.tex').write_text(body)
(root/'outputs/SWINGBY_AGGREGATE_PROPAGATION.tex').write_text(preamble+body+'\n\\end{document}\n')
old=(root/'outputs/main-10-review.tex').read_text()
assert old.count('\\end{document}') == 1
new=old.replace('\\end{document}', body+'\n\\end{document}')
labels=re.findall(r'\\label\{([^}]+)\}',new)
duplicates=sorted({x for x in labels if labels.count(x)>1})
assert not duplicates, duplicates
(root/'outputs/main-11-review.tex').write_text(new)
wrapper=(root/'work/appendix-check.tex').read_text()
(root/'work/aggregate-appendix-check.tex').write_text(wrapper.replace('\\end{document}',body+'\n\\end{document}'))
print('Wrote standalone and integrated TeX. No duplicate labels.')
