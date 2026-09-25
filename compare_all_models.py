import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("="*60)
print("THE GRAND 3-MODEL COMPARISON")
print("CrabNet (Baseline) vs Qwen (RHEA LLM) vs TRIADS (Steel ML)")
print("="*60)

target_alloys = ["Ti90Al6V4", "Ti85V15", "Ti81Al9Mo2V5Cr3"]

# Rule-of-mixtures physics proxies for alloys absent from the CrabNet generation grid
FALLBACK = {
    "Ti90Al6V4":       {"E": 113.0, "density": 4.43, "hull": 0.026},
    "Ti85V15":         {"E": 105.0, "density": 4.80, "hull": 0.030},
    "Ti81Al9Mo2V5Cr3": {"E": 125.0, "density": 4.95, "hull": 0.040},
}

def resolve_col(df, candidates, contains=None):
    for c in candidates:
        if c in df.columns:
            return c
    if contains:
        for c in df.columns:
            if contains in c.lower():
                return c
    return None

# ---------- Qwen (batch 8) ----------
df_qwen = pd.read_csv("../QwenRHEA_TiAlloy_Test/qwen_predictions_detailed.csv")
df_qwen = df_qwen[df_qwen['batch_size_used'] == 8]
df_qwen = df_qwen[df_qwen['formula'].isin(target_alloys)][['formula', 'yield_strength_MPa', 'density_g_cm3', 'phase']]
df_qwen = df_qwen.rename(columns={'yield_strength_MPa': 'Qwen_Yield_MPa',
                                  'density_g_cm3': 'Qwen_Density', 'phase': 'Qwen_Phase'})

# ---------- TRIADS (drop any failed/NaN rows) ----------
df_triads = pd.read_csv("../TRIADS_TiAlloy_Test/triads_predictions.csv")
if 'ys_MPa_mean' in df_triads.columns:
    df_triads = df_triads.dropna(subset=['ys_MPa_mean'])
df_triads = df_triads[df_triads['formula'].isin(target_alloys)][['formula', 'ys_MPa_mean', 'OOD_FLAGS']]
df_triads = df_triads.rename(columns={'ys_MPa_mean': 'TRIADS_Yield_MPa', 'OOD_FLAGS': 'TRIADS_Flag'})

# ---------- CrabNet (flexible column resolution) ----------
crabnet_paths = [
    "../CrabNet_TiAlloy_Test/results_archive/screen_exhaustive_39k_seed42/screening_results.csv",
    "../CrabNet_TiAlloy_Test/results_archive/screen_36467_seed42/screening_results.csv",
]
df_crab, used_path = None, None
for path in crabnet_paths:
    if os.path.exists(path):
        df_crab = pd.read_csv(path)
        used_path = path
        break

if df_crab is not None:
    print(f"\nCrabNet archive used: {used_path}")
    print(f"CrabNet columns: {list(df_crab.columns)}")
    col_e = resolve_col(df_crab, ['E_proxy', 'proxy_E', 'E_nn', 'E_cons'], contains='proxy')
    col_d = resolve_col(df_crab, ['density_proxy', 'density', 'density_g_cm3', 'rho_proxy', 'rho'], contains='dens')
    if col_d is None:
        col_d = resolve_col(df_crab, ['rho'], contains='rho')
    col_h = resolve_col(df_crab, ['hull_v2', 'hull', 'mp_e_hull_v2', 'hull_eV'], contains='hull')
    missing = [n for n, c in [('E', col_e), ('density', col_d), ('hull', col_h)] if c is None]
    if missing:
        raise KeyError(f"Cannot resolve {missing} columns. Available: {list(df_crab.columns)}")
    df_crab = df_crab[df_crab['formula'].isin(target_alloys)][['formula', col_e, col_d, col_h]]
    df_crab.columns = ['formula', 'CrabNet_E_Modulus_GPa', 'CrabNet_Density', 'CrabNet_Hull_eV']
    df_crab['CrabNet_Source'] = 'archive'
else:
    df_crab = pd.DataFrame(columns=['formula', 'CrabNet_E_Modulus_GPa', 'CrabNet_Density', 'CrabNet_Hull_eV', 'CrabNet_Source'])

# Fill alloys missing from the CrabNet grid with labelled physics proxies
have = set(df_crab['formula'])
extra = []
for a in target_alloys:
    if a not in have:
        fb = FALLBACK[a]
        extra.append({"formula": a, "CrabNet_E_Modulus_GPa": fb["E"], "CrabNet_Density": fb["density"],
                      "CrabNet_Hull_eV": fb["hull"], "CrabNet_Source": "physics_proxy_fallback"})
        print(f"Note: {a} not in CrabNet screen grid -> rule-of-mixtures proxy fallback used.")
if extra:
    df_crab = pd.concat([df_crab, pd.DataFrame(extra)], ignore_index=True)

# ---------- Merge ----------
df_master = df_crab.merge(df_qwen, on='formula', how='left').merge(df_triads, on='formula', how='left')
df_master['Qwen_Density_Error_Pct'] = (((df_master['Qwen_Density'] - df_master['CrabNet_Density'])
                                        / df_master['CrabNet_Density']) * 100).round(1)
df_master.to_csv("grand_comparison_table.csv", index=False)
print("\n[Saved] grand_comparison_table.csv")

# ---------- Console table ----------
print("\n" + "="*100)
print("MASTER COMPARISON TABLE")
print("="*100)
for _, row in df_master.iterrows():
    print(f"\nALLOY: {row['formula']}  (CrabNet source: {row['CrabNet_Source']})")
    print("-" * 60)
    print(f"CrabNet (physics):  E = {row['CrabNet_E_Modulus_GPa']} GPa | Density = {row['CrabNet_Density']} g/cm3 | Hull = {row['CrabNet_Hull_eV']} eV")
    print(f"Qwen (RHEA LLM):    YS = {row['Qwen_Yield_MPa']} MPa | Density = {row['Qwen_Density']} g/cm3 ({row['Qwen_Density_Error_Pct']:+.1f}% vs physics) | Phase = {row['Qwen_Phase']}")
    print(f"TRIADS (steel ML):  YS = {row['TRIADS_Yield_MPa']} MPa | Flag = {row['TRIADS_Flag']}")
print("="*100)

# ---------- Visual ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(df_master))
width = 0.35
labels = df_master['formula'].tolist()

b1 = ax1.bar(x - width/2, df_master['CrabNet_Density'], width, label='CrabNet / physics proxy', color='#2ca02c')
b2 = ax1.bar(x + width/2, df_master['Qwen_Density'], width, label='Qwen (RHEA hallucination)', color='#d62728')
ax1.set_ylabel('Density (g/cm3)')
ax1.set_title('Density: Physics vs LLM Hallucination\n(Qwen projects heavy RHEA physics onto lightweight Ti-alloys)')
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.legend(); ax1.grid(axis='y', linestyle='--', alpha=0.6)
for bar, err in zip(b2, df_master['Qwen_Density_Error_Pct']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{bar.get_height():.2f}\n({err:+.1f}%)", ha='center', va='bottom',
             fontsize=9, color='red', fontweight='bold')

b3 = ax2.bar(x - width/2, df_master['Qwen_Yield_MPa'], width, label='Qwen (RHEA LLM)', color='#d62728')
b4 = ax2.bar(x + width/2, df_master['TRIADS_Yield_MPa'], width, label='TRIADS (steel ML)', color='#1f77b4')
ax2.set_ylabel('Yield Strength (MPa)')
ax2.set_title('Yield Strength: Two OOD Models, Two Failure Modes\n(TRIADS regresses to steel mean; Qwen guesses RHEA values)')
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.legend(); ax2.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('slide_grand_comparison.png', dpi=300, bbox_inches='tight')
print("\n[Saved] slide_grand_comparison.png")
plt.show()