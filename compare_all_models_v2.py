import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, json, re

print("="*70); print("GRAND 3-MODEL COMPARISON v2 (full property audit)"); print("="*70)

# ---------- zero-parameter physics: rule-of-mixtures density ----------
AT_MASS = {"Ti":47.867,"Al":26.982,"V":50.942,"Mo":95.95,"Cr":51.996,
           "Fe":55.845,"Zr":91.224,"Sn":118.71}
EL_DENS = {"Ti":4.506,"Al":2.70,"V":6.11,"Mo":10.22,"Cr":7.19,
           "Fe":7.874,"Zr":6.52,"Sn":7.265}

def parse_formula(f):
    return {el: float(n) for el, n in re.findall(r'([A-Z][a-z]?)([\d\.]+)', f)}

def rom_density(f):
    at = parse_formula(f); tot = sum(at.values())
    w = {el: (n/tot)*AT_MASS[el] for el, n in at.items()}
    ws = sum(w.values())
    return 1.0 / sum((w[el]/ws)/EL_DENS[el] for el in w)

# ---------- load Qwen (batch 8) ----------
df_q = pd.read_csv("../QwenRHEA_TiAlloy_Test/qwen_predictions_detailed.csv")
df_q = df_q[df_q.batch_size_used == 8].copy()
df_q['rho_physics'] = df_q.formula.map(rom_density)
df_q['qwen_rho_err_pct'] = ((df_q.density_g_cm3 - df_q.rho_physics)/df_q.rho_physics*100).round(1)
df_q['YS_tabor_implied'] = (df_q.hardness_HV*9.80665/3).round(0)
df_q['tabor_inconsist_pct'] = ((df_q.YS_tabor_implied - df_q.yield_strength_MPa)/df_q.yield_strength_MPa*100).round(1)

# ---------- load TRIADS ----------
df_t = pd.read_csv("../TRIADS_TiAlloy_Test/triads_predictions.csv")
if 'ys_MPa_mean' in df_t.columns: df_t = df_t.dropna(subset=['ys_MPa_mean'])

# ---------- load CrabNet archive ----------
crab = "../CrabNet_TiAlloy_Test/results_archive/screen_exhaustive_39k_seed42/screening_results.csv"
df_c = pd.read_csv(crab) if os.path.exists(crab) else pd.DataFrame()
keep = [c for c in ['formula','E_nn','K_nn','G_nn','rho_proxy','hull_v2','pugh_proxy','spec_stiff_cons'] if c in df_c.columns]
df_c = df_c[keep] if keep else pd.DataFrame(columns=['formula'])

# ---------- master table (all 20 Qwen alloys) ----------
m = df_q[['formula','rho_physics','density_g_cm3','qwen_rho_err_pct','yield_strength_MPa',
          'hardness_HV','YS_tabor_implied','tabor_inconsist_pct','phase']].copy()
m = m.rename(columns={'density_g_cm3':'qwen_density','yield_strength_MPa':'qwen_YS',
                      'hardness_HV':'qwen_HV','phase':'qwen_phase'})
m = m.merge(df_c, on='formula', how='left')
m = m.merge(df_t[['formula','ys_MPa_mean','OOD_FLAGS']].rename(
            columns={'ys_MPa_mean':'triads_YS','OOD_FLAGS':'triads_flag'}), on='formula', how='left')
m.to_csv("grand_comparison_table_v2.csv", index=False)
print("[Saved] grand_comparison_table_v2.csv  (20 alloys)")
print(f"Qwen density error: mean {m.qwen_rho_err_pct.mean():+.1f}%, max {m.qwen_rho_err_pct.max():+.1f}%")
print(f"Qwen Tabor self-inconsistency: mean {m.tabor_inconsist_pct.abs().mean():.0f}%")

# ---------- figure ----------
fig, ax = plt.subplots(2, 2, figsize=(17, 10))

# (0,0) coverage heatmap
props = ["E/K/G moduli","Density","Hull/stability","Pugh/spec.stiff","Phase","Yield strength","Hardness"]
cov = np.array([
    [1,0,0],[1,1,0],[1,0,0],[1,0,0],[0,1,0],[0,1,1],[0,1,0]])
im = ax[0,0].imshow(cov, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax[0,0].set_xticks(range(3)); ax[0,0].set_xticklabels(['CrabNet','Qwen','TRIADS'])
ax[0,0].set_yticks(range(len(props))); ax[0,0].set_yticklabels(props, fontsize=9)
for i in range(len(props)):
    for j in range(3):
        ax[0,0].text(j, i, '✔' if cov[i,j] else '✘', ha='center', va='center', fontsize=12)
ax[0,0].set_title('Property coverage: sets are disjoint by design\n(no property is predicted by all three)')

# (0,1) density error across all 20 alloys
s = m.sort_values('qwen_rho_err_pct')
ax[0,1].barh(s.formula, s.qwen_rho_err_pct, color='#d62728')
ax[0,1].axvline(0, color='k', lw=1)
ax[0,1].set_xlabel('Qwen density error vs rule-of-mixtures physics (%)')
ax[0,1].set_title(f'Qwen density hallucination, all 20 alloys\n(mean {m.qwen_rho_err_pct.mean():+.1f}%, max {m.qwen_rho_err_pct.max():+.1f}%)')
ax[0,1].tick_params(labelsize=8); ax[0,1].grid(axis='x', ls='--', alpha=.5)

# (1,0) YS: Qwen vs TRIADS vs Qwen's own Tabor-implied YS
ov = m.dropna(subset=['triads_YS'])
x = np.arange(len(ov)); w = 0.27
ax[1,0].bar(x-w, ov.qwen_YS, w, label='Qwen stated YS', color='#d62728')
ax[1,0].bar(x,   ov.YS_tabor_implied, w, label='YS implied by Qwen hardness (Tabor H≈3σy)', color='#ff9896')
ax[1,0].bar(x+w, ov.triads_YS, w, label='TRIADS (steel-mean regression)', color='#1f77b4')
ax[1,0].set_xticks(x); ax[1,0].set_xticklabels(ov.formula, fontsize=9)
ax[1,0].set_ylabel('Yield strength (MPa)'); ax[1,0].legend(fontsize=8)
ax[1,0].set_title('Yield strength: OOD disagreement + Qwen self-inconsistency')
ax[1,0].grid(axis='y', ls='--', alpha=.5)

# (1,1) throughput, read dynamically from comparison JSONs
tps, names = [], []
for jf, key in [("model_results_crabnet.json", None),
                ("model_results_qwenrhea.json", None),
                ("model_results_triads.json", None)]:
    if os.path.exists(jf):
        d = json.load(open(jf))
        if 'throughput_samples_per_s' in d:
            v = d['throughput_samples_per_s']
            tps.append(max(v.values()) if isinstance(v, dict) else v)
        elif 'e2e_throughput_samples_per_s' in d:
            tps.append(max(d['e2e_throughput_samples_per_s'].values()))
        else:
            tps.append(d.get('throughput', {}).get('samples_per_s', np.nan))
        names.append(d.get('model', jf)[:12])
if tps:
    ax[1,1].bar(names, tps, color=['#2ca02c','#d62728','#1f77b4'][:len(tps)])
    ax[1,1].set_yscale('log')
    for i, v in enumerate(tps):
        ax[1,1].text(i, v*1.2, f'{v:,.1f}', ha='center', fontweight='bold', fontsize=9)
    ax[1,1].set_ylabel('E2E throughput (samples/s, log)')
    ax[1,1].set_title('Production speed: CrabNet ≫ TRIADS  Qwen')
    ax[1,1].grid(axis='y', ls='--', alpha=.5)

plt.tight_layout()
plt.savefig('slide_grand_comparison_v2.png', dpi=300, bbox_inches='tight')
print("[Saved] slide_grand_comparison_v2.png")
plt.show()