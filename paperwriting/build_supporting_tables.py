#!/usr/bin/env python3
"""Rebuild final-layer ceiling tables from saved benchmark inputs; never refit results."""
from pathlib import Path
import ast
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paperwriting/sections'
AUDIT = ROOT / 'paperwriting/supporting_data'

def literal(path, key):
    for node in ast.parse((ROOT / path).read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == key for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and getattr(node.target, 'id', '') == key:
            return ast.literal_eval(node.value)
    raise KeyError(key)

def rows(path):
    with (ROOT / path).open() as f:
        return list(csv.DictReader(f))

def number(x):
    try: return float(x)
    except (ValueError, TypeError): return float('nan')

def fmt(x, digits=3):
    return f'{x:.{digits}f}' if math.isfinite(x) else '--'

def sci(x):
    if not math.isfinite(x): return '--'
    a, b = f'{x:.2e}'.split('e')
    return rf'${a}\times10^{{{int(b)}}}$'

def dump(name, data):
    AUDIT.mkdir(exist_ok=True)
    with (AUDIT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def table(name, caption, label, spec, headers, data):
    n = len(headers)
    line = ' & '.join(headers) + r' \\'
    text = [r'{\small\setlength{\tabcolsep}{4pt}',rf'\begin{{longtable}}{{{spec}}}',
            rf'\caption{{{caption}}}\label{{{label}}} \\',r'\toprule',line,r'\midrule',r'\endfirsthead',
            rf'\multicolumn{{{n}}}{{c}}{{\tablename\ \thetable{{}} -- continued}} \\',r'\toprule',line,
            r'\midrule',r'\endhead',rf'\midrule\multicolumn{{{n}}}{{r}}{{Continued on next page}} \\',
            r'\endfoot',r'\bottomrule',r'\endlastfoot']
    text += [' & '.join(r) + r' \\' for r in data]
    text += [r'\end{longtable}', '}']
    (OUT / name).write_text('\n'.join(text)+'\n')

LABEL = {'geco':'GECO FP','geco_gp':'GECO GP','meco':'MECO FP','meco_gp':'MECO GP','meco_sp':'MECO SP',
'provo':'Provo FP','provo_gp':'Provo GP','provo_sp':'Provo SP','onestop_fp':'OneStop FP','onestop_gp':'OneStop GP','onestop_sp':'OneStop SP',
'naturalstories_spr':'Natural Stories SPR','ucl_reading':'UCL SPR','zuco_reading':'ZuCo NR','zuco_reading_sr':'ZuCo SR','zuco_reading_tsr':'ZuCo TSR',
'lpp_fmri':'Le Petit Prince fMRI','ns_fmri':'Natural Stories fMRI','pereira_fmri':'Pereira fMRI','tuckute_fmri':'Tuckute fMRI',
'fedorenko2016_ecog':'Fedorenko intracranial','podcast_ieeg':'Podcast ECoG','broderick':'Broderick EEG','megmasc':'MEG-MASC',
'nieuwland':'Nieuwland N400','szewczyk_f2007':'Federmeier 2007 N400','szewczyk_h2019':'Hubbard 2019 N400'}
RDM_LABEL = {**LABEL,'pereira':'Pereira fMRI','tuckute':'Tuckute fMRI','podcast':'Podcast ECoG','geco':'GECO','meco':'MECO','provo':'Provo',
'onestop':'OneStop','naturalstories':'Natural Stories','ucl_reading':'UCL','zuco_reading':'ZuCo'}
UNITS = literal('scripts/analysis/matched_scale_regression.py', '_SUMMARY_UNITS')
TAGS = literal('neuroai/analysis/benchmark_common.py', 'DLL_CEILING_TAGS')

def build_direct():
    for track, units in UNITS.items():
        audit, rendered = [], []
        for slug, ds in units.items():
            source = f'outputs/benchmark_summary/{slug}/all_models_scores.csv'
            data = rows(source)
            cs = {number(r.get('ceiling_r')) for r in data}
            assert len(cs) == 1, (slug, cs)
            c = cs.pop()
            assert c > 0
            valid = [r for r in data if math.isfinite(number(r.get('score_last_norm')))]
            for r in valid:
                raw = number(r.get('score_last'))
                if math.isfinite(raw):
                    assert math.isclose(raw/c, number(r['score_last_norm']), rel_tol=1e-6, abs_tol=1e-8), slug
            ns = [number(r['score_last_norm']) for r in valid]
            above = sum(v > 1 for v in ns)
            rec = dict(dataset=ds, source=source, reference=c, n_models=len(ns), max_normalized=max(ns), n_above=above)
            if track == 'embedding':
                kind = 'Rank' if ds in ['fedorenko2016_ecog','nieuwland','szewczyk_f2007','szewczyk_h2019'] else ('ISC' if ds=='lpp_fmri' else 'Pearson')
                rec['reference_type'] = kind
                rendered.append([LABEL[ds],fmt(c),kind,str(len(ns)),fmt(max(ns)),f'{above}/{len(ns)}'])
            else:
                jf = ROOT / 'outputs/noise_ceilings' / f'delta_loglik_ceiling_{TAGS[slug]}.json'
                j = json.loads(jf.read_text())
                assert math.isclose(c,float(j['ceiling_delta_loglik_per_obs']),rel_tol=1e-8)
                sd = number(j.get('ceiling_delta_loglik_per_obs_sd'))
                used = j.get('n_splits_converged',j.get('n_iterations_used'))
                requested = j.get('n_splits_requested',j.get('n_iterations_requested'))
                splits = f'{used}/{requested}' if used is not None and requested is not None else '--'
                if isinstance(j.get('n_splits_used'),dict):
                    v = list(j['n_splits_used'].values())
                    splits = f'{min(v)}--{max(v)}/{requested}' if min(v)!=max(v) else f'{v[0]}/{requested}'
                rec.update(reference_source=str(jf.relative_to(ROOT)),split_sd=sd,split_count=splits)
                flag = (math.isfinite(sd) and sd/c>.5) or above/len(ns)>.1
                rendered.append([LABEL[ds]+(r'$\dagger$' if flag else ''),sci(c),sci(sd),splits,f'{above}/{len(ns)}',fmt(max(ns))])
            audit.append(rec)
        dump(f'{track}_ceiling_table.csv',audit)
        if track=='embedding':
            table('corr_ceiling_table.tex',r'Correlation references used to normalize the final-layer embedding scores. Maximum is the largest normalized score among models with a finite result, not a selected layer. Pearson and Rank indicate Pearson- and Spearman-based split-half references; ISC denotes the precomputed intersubject-correlation conversion. Above counts normalized scores greater than one. Rank references are reliability normalizers, not exact Pearson ceilings. See Appendix~\ref{app:corr-ceiling} for aggregation and target-matching qualifications.',
                  'tab:corr-ceilings','@{}lrlrrr@{}',['Dataset / measure','Reference','Type','$N$','Maximum','Above'],rendered)
        else:
            table('dll_ceiling_table.tex',r'Split-half likelihood references used in final-layer surprisal normalization, in $\Delta\ell$ per observation. SD is variability across splits or iterations, not a standard error; -- means the aggregate file supplies no such value. Splits reports successful/attempted draws where available. Above is the number of finite model scores exceeding the reference, and Maximum is the largest normalized score. A dagger flags split SD/reference $>.5$ or more than 10\% of models above one; this is a descriptive diagnostic, not an exclusion rule. Pereira split counts apply to each of its two ROIs; region-specific references are detailed in the text.',
                  'tab:dll-ceilings','@{}lrrrrr@{}',['Dataset / measure','Reference','SD','Splits','Above','Maximum'],rendered)

def build_rdm():
    data=rows('reports/rsa_summary/rsa_pareto/rsa_ratios_embedding.csv')
    sur=rows('reports/rsa_summary/rsa_pareto/rsa_ratios_surprisal.csv')
    groups={}; sg={}
    for r in data:groups.setdefault((r['dataset'],r['unit']),[]).append(r)
    for r in sur:sg.setdefault((r['dataset'],r['unit']),[]).append(r)
    out=[]; audit=[]
    for (ds,u),rs in groups.items():
        c={number(r['ceiling']) for r in rs};assert len(c)==1
        c=c.pop();nunit={r['n_units'] for r in rs};assert len(nunit)==1
        nu=nunit.pop();nu=str(int(float(nu))) if nu else '1'
        title=RDM_LABEL[ds]+(' / '+u if u!='all' else '')
        em=max(number(r['norm']) for r in rs);sm=max(number(r['norm']) for r in sg[(ds,u)])
        out.append([title,fmt(c),nu,fmt(em),fmt(sm)])
        audit.append(dict(dataset=ds,unit=u,reference_summary=c,n_usable_units=nu,max_embedding=em,max_surprisal=sm))
    dump('rsa_ceiling_table.csv',audit)
    table('rsa_ceiling_table.tex',r'RSA-correlation references and maximum final-layer normalized scores. For neural datasets, Reference is the median of the unit-level ceilings, reported for context; normalization is performed within units before taking the median and is not division by this displayed summary. Units counts usable positive-ceiling response units. For behavioral measures there is one ceiling per measure. Maximum columns are maxima across models of the dataset/measure score.',
          'tab:rsa-ceilings','@{}lrrrr@{}',['Dataset / measure','Reference','Units','Max. emb.','Max. sur.'],out)
    cs=rows('reports/rsa_summary/rdm_regression/rdm_ceiling_values.csv')
    out=[];audit=[]
    for r in cs:
        ds,u=r['dataset_key'],r['unit'];c=number(r['ceiling_dll'])
        out.append([RDM_LABEL[ds]+(' / '+u if u!='all' else ''),sci(c),'Yes' if r['excluded_from_pareto'].lower()=='false' else 'No'])
        audit.append(dict(dataset=ds,unit=u,reference=c,included=r['excluded_from_pareto'].lower()=='false'))
    dump('rdm_dll_ceiling_table.csv',audit)
    table('rdm_dll_ceiling_table.tex',r'Dataset/measure-level split-half likelihood references for the RDM-regression fronts in Figure~\ref{fig:rdm-pareto}, in $\Delta\ell$ per pair. They differ from the unit-wise normalization of the RSA correlations. All listed references are positive.',
          'tab:rdm-dll-ceilings','@{}lrc@{}',['Dataset / measure','Reference','Included'],out)

GROUPS = ['LLM','Memory-Constrained','BabyLM','Constrained-BabyLM']
BEHAVIORAL = {'geco','geco_gp','meco','meco_gp','meco_sp','provo','provo_gp','provo_sp','onestop_fp','onestop_gp',
              'onestop_sp','naturalstories_spr','ucl_reading','zuco_reading','zuco_reading_sr','zuco_reading_tsr'}

def build_group_medians():
    """Unadjusted group medians of BehavScore/BrainScore (median over units of the last-layer normalized score)."""
    cells, audit = {}, []
    for track, units in UNITS.items():
        per_model = {}
        for slug, ds in units.items():
            for r in rows(f'outputs/benchmark_summary/{slug}/all_models_scores.csv'):
                v = number(r.get('score_last_norm'))
                if not math.isfinite(v): continue
                m = per_model.setdefault(r['model_id'], {'group': r['group'], 'beh': [], 'neu': []})
                m['beh' if ds in BEHAVIORAL else 'neu'].append(v)
        for g in GROUPS:
            ms = [m for m in per_model.values() if m['group'] == g]
            for dom in ('beh', 'neu'):
                vals = [statistics.median(m[dom]) for m in ms if m[dom]]
                med = statistics.median(vals) if vals else float('nan')
                cells[(track, dom, g)] = (med, len(vals))
                audit.append(dict(track=track, domain=dom, group=g, n_models=len(vals), median=med))
    dump('group_medians.csv', audit)
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{Unadjusted group medians of the composite scores (median across models within a group of each model' "'" r's BehavScore or BrainScore; last layer, ceiling-normalized). $N$ is the number of models with at least one finite score. These raw differences confound the constraints with size and perplexity; Table~\ref{tab:main-regression} adjusts for both.}',
             r'\label{tab:medians}', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'& & \multicolumn{2}{c}{Embeddings} & \multicolumn{2}{c}{Surprisal}\\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
             r'Group & $N$ & BehavScore & BrainScore & BehavScore & BrainScore\\', r'\midrule']
    for g in GROUPS:
        n = cells[('embedding', 'beh', g)][1]
        vals = [fmt(cells[(t, d, g)][0]) for t in ('embedding', 'surprisal') for d in ('beh', 'neu')]
        lines.append(' & '.join(['Standard LLMs' if g == 'LLM' else g, str(n)] + vals) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (OUT / 'results_medians_table.tex').write_text('\n'.join(lines) + '\n')

RI = 'reports/robust_inference/coefficients.csv'
RI_TERMS = [('M', 'Memory only'), ('D', 'Data only'), ('M:D', r'Memory $\times$ data'), ('both', 'Both constraints'),
            ('size', r'$\log_{10}$ parameters'), ('ppl', r'$\log_{10}$ perplexity')]

def _stars(p):
    return '' if not math.isfinite(p) else ('^{***}' if p < .001 else '^{**}' if p < .01 else '^{*}' if p < .05 else '')

def _cell(r, ci=False):
    if r is None: return '--'
    e, p = number(r['est']), number(r['p'])
    if ci: return rf"${e:+.3f}\ [{number(r['lo']):+.3f}, {number(r['hi']):+.3f}]{_stars(p)}$".replace('+0.', '.').replace('-0.', '-.').replace('+.', '.')
    return rf"${e:+.3f}\ ({number(r['se']):.3f}){_stars(p)}$".replace('+0.', '.').replace('-0.', '-.').replace('+.', '.').replace('(0.', '(.')

def _short(x):
    return f'${x:+.3f}$'.replace('+0.', '.').replace('-0.', '-.')

def build_robust_tables():
    """Table 2 (preferred S2 fits + joint domain differences) and the appendix robustness table, from
    scripts/analysis/robust_inference.R output."""
    co = rows(RI)
    def get(track, analysis, variant, domain, term):
        for r in co:
            if (r['track'], r['analysis'], r['domain'], r['term']) == (track, analysis, domain, term) and \
               (variant is None or r['variant'] == variant):
                return r
        return None
    n = {}
    for track in ('embedding', 'surprisal'):
        for dom in ('behavioral', 'neural'):
            d = [x for x in rows(f'reports/robust_inference/rows_{track}.csv') if x['modality'] == dom]
            n[(track, dom)] = (len(d), len({x['dataset'] for x in d}))
    # Table 2
    body = []
    for t, lab in RI_TERMS:
        cells = []
        for track in ('embedding', 'surprisal'):
            cells += [_cell(get(track, 'ladder', 'S2', 'behavioral', t)), _cell(get(track, 'ladder', 'S2', 'neural', t)),
                      _cell(get(track, 'joint', 'ceiling-normalized', 'neural - behavioral', t))]
        body.append(' & '.join([lab] + cells) + r' \\')
        if t == 'both': body.append(r'\midrule')
    nrow = ' & '.join(['Rows / dataset units'] + [f'{n[(tr, d)][0]:,} / {n[(tr, d)][1]}' if d else 'joint'
                                                  for tr in ('embedding', 'surprisal') for d in ('behavioral', 'neural', None)])
    tex = [r'\begin{table}[t]', r'\centering', r'\small',
           r'\caption{Group associations in final-layer direct prediction (Equation~\ref{eq:group}, preferred specification). Coefficients are in noise-ceiling-normalized score units with standard errors in parentheses (Satterthwaite degrees of freedom). Behavioral and neural columns are separate fits with random intercepts for dataset unit, model lineage, model and corpus, and dataset-unit random slopes for size and perplexity. The Neural $-$ behavioral columns come from a joint model of both domains with domain interactions and directly test whether a coefficient differs between them. Single-constraint and combined contrasts are relative to Standard LLMs conditional on parameter count and perplexity. $^*p<.05$, $^{**}p<.01$, $^{***}p<.001$. Robustness checks are in Table~\ref{tab:robustness}.}',
           r'\label{tab:main-regression}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lrrrrrr}', r'\toprule',
           r'& \multicolumn{3}{c}{Embeddings} & \multicolumn{3}{c}{Surprisal}\\',
           r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
           r'& Behavioral & Neural & Neural $-$ behav. & Behavioral & Neural & Neural $-$ behav.\\', r'\midrule',
           *body, r'\midrule', nrow + r' \\', r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    (OUT / 'results_main_table.tex').write_text('\n'.join(tex) + '\n')
    # Appendix robustness table
    cols = [('ladder', 'S0', 'S0'), ('ladder', 'S2', 'S2 (Table 2)'), ('lineage_bootstrap', None, 'Lineage bootstrap'),
            ('robustness', 'no_dagger', r'No $\dagger$ units')]
    out = []
    for track in ('embedding', 'surprisal'):
        c = cols if track == 'surprisal' else cols[:3] + [None]
        for dom in ('behavioral', 'neural'):
            out.append(rf'\multicolumn{{6}}{{l}}{{\textit{{{track.capitalize()}, {dom}}}}} \\')
            for t, lab in RI_TERMS[:4]:
                lo = [r for r in co if (r['track'], r['domain'], r['term'], r['analysis']) == (track, dom, t, 'leave_one_unit_out')]
                rng = (_short(min(number(r['est']) for r in lo)) + ' to ' + _short(max(number(r['est']) for r in lo))
                       + rf" ({sum(number(r['p']) < .05 for r in lo)}/{len(lo)})") if lo else '--'
                out.append(' & '.join([lab] + [_cell(get(track, *x[:2], dom, t)) if x else '--' for x in c] + [rng]) + r' \\')
    head = ' & '.join(['Term'] + [x[2] for x in cols] + ['Leave one unit out']) + r' \\'
    tex = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
           r'\caption{Robustness of the constraint contrasts. Cells are coefficients (SE) from separate refits. S0 has a dataset-unit random intercept only and treats rows as otherwise independent; S2 is the preferred specification of Table~\ref{tab:main-regression}. Lineage bootstrap: S0 refit with model lineages resampled within constraint groups (999 resamples; SE is the bootstrap SD, $p$ two-sided). No $\dagger$ units: S2 without the four units whose likelihood references are flagged in Table~\ref{tab:dll-ceilings} (surprisal only). Leave one unit out: range of S2 estimates over refits that each omit one dataset unit, with the number reaching $p<.05$. Refits with winsorized scores, without the GPT-BERT models, or with depth as a covariate change no conclusion (Section~\ref{app:robustness}).}',
           r'\label{tab:robustness}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{llllll}', r'\toprule', head, r'\midrule',
           *out, r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    (OUT / 'robustness_table.tex').write_text('\n'.join(tex) + '\n')

RSA_ORDER = ['Pereira · fMRI', 'Tuckute B · fMRI', 'Broderick · EEG', 'MEG-MASC · MEG', 'Nieuwland · N400',
             'Federmeier et al. (2007) · N400', 'Hubbard et al. (2019) · N400', 'Podcast · ECoG (−0.2–0.8 s)',
             'Provo · FP', 'Provo · GP', 'Provo · SP', 'MECO English · FP', 'MECO English · GP', 'MECO English · SP',
             'GECO · FP', 'GECO · GP', 'OneStop · FP', 'OneStop · GP', 'OneStop · SP', 'Natural Stories · SPR',
             'UCL Reading · SPR', 'ZuCo · NR', 'ZuCo · SR', 'ZuCo · TSR']

def build_rsa_summary():
    """RSA size and agreement correlations across the 118 models, from reports/rsa_summary/rsa/model_scores{,_dll}.csv
    (embedding, surprisal). Groups are reassigned with the current registry rule; the within-group size correlation is
    the partial correlation of RSA with log10 parameters controlling for constraint group."""
    import numpy as np, pandas as pd
    import sys; sys.path.insert(0, str(ROOT))
    from neuroai.analysis.benchmark_common import family_group
    def resid(y, G):
        b, *_ = np.linalg.lstsq(G, y, rcond=None); return y - G @ b
    res, audit = {}, []
    for track, f in (('emb', 'model_scores'), ('sur', 'model_scores_dll')):
        d = pd.read_csv(ROOT / f'reports/rsa_summary/rsa/{f}.csv')
        d['group'] = [family_group(a, m) for a, m in zip(d['family'], d['model_id'])]
        d['ls'] = np.log10(d['n_params'])
        for title, g in d.groupby('title'):
            g = g.dropna(subset=['rsa_pearson', 'ls'])
            G = pd.get_dummies(g['group']).to_numpy(float)
            size = np.corrcoef(g['rsa_pearson'], g['ls'])[0, 1]
            within = np.corrcoef(resid(g['rsa_pearson'].to_numpy(), G), resid(g['ls'].to_numpy(), G))[0, 1]
            h = g.dropna(subset=['benchmark_score'])
            agree = np.corrcoef(h['rsa_pearson'], h['benchmark_score'])[0, 1]
            res[(title, track)] = (size, within, agree)
            audit.append(dict(title=title, track=track, n=len(g), size_r=size, size_r_within_group=within, agreement_r=agree))
    dump('rsa_summary_table.csv', audit)
    lines = [r'\begin{table}[p]', r'\centering\small',
             r'\caption{Descriptive Pearson correlations across 118 models: RSA with $\log_{10}$ parameter count (size), the same correlation within constraint groups (partial correlation controlling for group membership, so that it does not merely reflect that the smallest models are the constrained ones), and RSA with the corresponding direct-prediction score (agreement: embedding RSA with encoding $r$, surprisal RSA with surprisal $\Delta\ell$). Emb.\ denotes embeddings; Sur.\ denotes surprisal. No model-family adjustment or significance test is applied.}',
             r'\label{tab:rsa-summary}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'& \multicolumn{2}{c}{Size} & \multicolumn{2}{c}{Size within group} & \multicolumn{2}{c}{Agreement}\\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
             r'Dataset / measure & Emb. & Sur. & Emb. & Sur. & Emb. & Sur.\\', r'\midrule']
    for i, t in enumerate(RSA_ORDER):
        if i == 8: lines.append(r'\midrule')
        e, s_ = res[(t, 'emb')], res[(t, 'sur')]
        name = t.replace('·', '/').replace('–', '--').replace(' (−0.2--0.8 s)', '')
        lines.append(' & '.join([name] + [fmt(v) for v in (e[0], s_[0], e[1], s_[1], e[2], s_[2])]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (OUT / 'rsa_summary_table.tex').write_text('\n'.join(lines) + '\n')

def build_stepb_table():
    """Design-based comparisons (scripts/analysis/robust_inference_stepB.R): within-architecture data dose-response,
    FoX matched pairs, and memory contrasts per mechanism."""
    co = rows('reports/robust_inference/stepB_coefficients.csv')
    def get(track, dom, analysis, term):
        for r in co:
            if (r['track'], r['domain'], r['analysis'], r['term']) == (track, dom, analysis, term): return r
        return None
    spec = [(None, r'\textit{Within architecture (36 models; reference: Pile)}'),
            (('dose_response', 'BabyLM 100M - Pile'), 'BabyLM 100M'), (('dose_response', 'BabyLM 10M - Pile'), 'BabyLM 10M'),
            (None, r'\textit{Matched pairs (2 pairs, same data and size)}'),
            (('fox_pairs', 'FoX - softmax baseline'), r'Forget gate $-$ softmax baseline'),
            (None, r'\textit{Memory only, by mechanism (vs Standard LLMs, preferred specification)}'),
            (('mechanism', 'distance bias: memory only'), 'Distance bias (ALiBi, BLOOM)'),
            (('mechanism', 'forget gate: memory only'), 'Forget gate (Forgetting, FoX)'),
            (('mechanism', 'stick-breaking: memory only'), r'Stick-breaking (incl.\ MoLM)'),
            (('mechanism', 'hard window: memory only'), r'Hard $n$-gram window'),
            (('mechanism', 'CBR-RNN: memory only'), 'CBR-RNN')]
    body = []
    for key, lab in spec:
        if key is None:
            body.append(rf'\multicolumn{{5}}{{l}}{{{lab}}} \\'); continue
        body.append(' & '.join([r'\quad ' + lab] + [_cell(get(t, d, *key)) for t in ('embedding', 'surprisal')
                                                     for d in ('behavioral', 'neural')]) + r' \\')
    tex = [r'\begin{table}[t]', r'\centering', r'\small',
           r'\caption{Design-based comparisons. Within architecture: the six memory-limited architectures of \citet{bu2026staticlinearattentionconstraints}, each trained on BabyLM 10M, BabyLM 100M and the Pile at 2 and 4 layers, so that architecture, code and size are fixed and only the training data differ (random intercepts for architecture, model and unit; no size or perplexity adjustment, since perplexity is an outcome of the data). Matched pairs: the two Forgetting Transformer models against their own softmax-attention baselines (mean difference across dataset units; $p$ from a $t$ test across units). By mechanism: the memory-only contrast of Table~\ref{tab:main-regression} estimated separately for each memory mechanism. Coefficients (SE) in normalized score units; $^*p<.05$, $^{**}p<.01$, $^{***}p<.001$.}',
           r'\label{tab:design}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lrrrr}', r'\toprule',
           r'& \multicolumn{2}{c}{Embeddings} & \multicolumn{2}{c}{Surprisal}\\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
           r'& Behavioral & Neural & Behavioral & Neural\\', r'\midrule', *body, r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    (OUT / 'design_table.tex').write_text('\n'.join(tex) + '\n')

if __name__ == '__main__':
    build_direct();build_rdm();build_group_medians();build_robust_tables();build_rsa_summary();build_stepb_table()
    print('Rebuilt four ceiling tables, the group-median, main regression and robustness tables and their source-value CSVs; verified direct-score denominators.')
