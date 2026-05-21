import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


# 1. LOAD DATA

results = {}
merged = {}

for year in ['2016', '2020', '2024']:
    results[year] = pd.read_csv(f'results_{year}.csv')
    merged[year]  = pd.read_csv(f'merged_{year}.csv')


# 2. CLEAN: drop ERROR rows

for year in results:
    before = len(results[year])
    results[year] = results[year][results[year]['label'] != 'ERROR'].copy()
    after = len(results[year])
    print(f"{year}: dropped {before - after} ERROR rows ({after} remaining)")


# 3. EXTRACT OUTLET FROM URL (merged files)

def extract_outlet(url):
    url = str(url).lower()
    if 'foxnews' in url:   return 'Fox News'
    if 'apnews' in url:    return 'AP'
    if 'cnn' in url:       return 'CNN'
    if 'nytimes' in url:   return 'NYT'
    return 'Other'

for year in merged:
    merged[year]['outlet'] = merged[year]['SOURCEURL'].apply(extract_outlet)


# 4. BERT SENTIMENT: normalize to percentages

# Combine all years, tag with year
bert_all = []
for year, df in results.items():
    df = df.copy()
    df['year'] = year
    bert_all.append(df)
bert_all = pd.concat(bert_all, ignore_index=True)

# Count labels per outlet per year
bert_counts = (
    bert_all
    .groupby(['year', 'outlet', 'label'])
    .size()
    .reset_index(name='count')
)

# Pivot and normalize to %
bert_pivot = bert_counts.pivot_table(
    index=['year', 'outlet'], columns='label', values='count', fill_value=0
).reset_index()

label_cols = [c for c in ['negative', 'neutral', 'positive'] if c in bert_pivot.columns]
bert_pivot['total'] = bert_pivot[label_cols].sum(axis=1)
for col in label_cols:
    bert_pivot[f'{col}_pct'] = bert_pivot[col] / bert_pivot['total'] * 100

print("\nBERT sentiment % per outlet per year:")
print(bert_pivot[['year', 'outlet', 'total'] + [f'{c}_pct' for c in label_cols]].to_string(index=False))


# 5. OUTLET PROFILES (across all years)

outlet_profile = (
    bert_all
    .groupby(['outlet', 'label'])
    .size()
    .reset_index(name='count')
)
outlet_profile_pivot = outlet_profile.pivot_table(
    index='outlet', columns='label', values='count', fill_value=0
).reset_index()

label_cols_p = [c for c in ['negative', 'neutral', 'positive'] if c in outlet_profile_pivot.columns]
outlet_profile_pivot['total'] = outlet_profile_pivot[label_cols_p].sum(axis=1)
for col in label_cols_p:
    outlet_profile_pivot[f'{col}_pct'] = outlet_profile_pivot[col] / outlet_profile_pivot['total'] * 100

print("\nOverall outlet profiles (all years combined):")
print(outlet_profile_pivot[['outlet', 'total'] + [f'{c}_pct' for c in label_cols_p]].to_string(index=False))


# 6. GDELT AvgTone: per outlet per year

gdelt_all = []
for year, df in merged.items():
    df = df.copy()
    df['year'] = year
    gdelt_all.append(df[['year', 'outlet', 'AvgTone']])
gdelt_all = pd.concat(gdelt_all, ignore_index=True)

gdelt_summary = (
    gdelt_all[gdelt_all['outlet'] != 'Other']
    .groupby(['year', 'outlet'])['AvgTone']
    .agg(['mean', 'count'])
    .reset_index()
    .rename(columns={'mean': 'avg_tone', 'count': 'n_articles'})
)

print("\nGDELT AvgTone per outlet per year:")
print(gdelt_summary.to_string(index=False))


# 7. COMPARE BERT vs GDELT
#    Map BERT % positive/negative to a net score: positive% - negative%
#    Compare with GDELT AvgTone direction

bert_net = bert_pivot[['year', 'outlet', 'positive_pct', 'negative_pct']].copy()
bert_net['bert_net_score'] = bert_net['positive_pct'] - bert_net['negative_pct']

comparison = bert_net.merge(gdelt_summary, on=['year', 'outlet'], how='inner')

print("\nBERT net score vs GDELT AvgTone:")
print(comparison[['year', 'outlet', 'bert_net_score', 'avg_tone']].to_string(index=False))


# 8. VISUALIZATIONS

OUTLETS   = ['CNN', 'Fox News', 'NYT']
YEARS     = ['2016', '2020', '2024']
COLORS    = {'negative': '#d62728', 'neutral': '#aec7e8', 'positive': '#2ca02c'}

# Fig 1: BERT sentiment % per outlet per year (stacked bar)
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig.suptitle("Trump Coverage Sentiment by Outlet (BERT)", fontsize=14, fontweight='bold')

for ax, year in zip(axes, YEARS):
    df_y = bert_pivot[bert_pivot['year'] == year].set_index('outlet').reindex(OUTLETS)
    bottom = np.zeros(len(OUTLETS))
    bar_data = {}
    for label in ['negative', 'neutral', 'positive']:
        col = f'{label}_pct'
        vals = df_y[col].fillna(0).values if col in df_y.columns else np.zeros(len(OUTLETS))
        bars = ax.bar(OUTLETS, vals, bottom=bottom, label=label.capitalize(),
                      color=COLORS[label], edgecolor='white', linewidth=0.5)
        bar_data[label] = (vals, bottom.copy())
        # Add % label inside bar segment if segment is wide enough
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 8:
                ax.text(i, b + v / 2, f'{v:.0f}%', ha='center', va='center',
                        fontsize=7.5, color='white', fontweight='bold')
        bottom += vals

    ax.set_title(f'{year} Election', fontsize=11)
    ax.set_xlabel('Outlet')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 112)
    ax.set_xticks(range(len(OUTLETS)))
    ax.set_xticklabels(OUTLETS, fontsize=9)

    # Sample size above bar
    for i, outlet in enumerate(OUTLETS):
        row = bert_pivot[(bert_pivot['year'] == year) & (bert_pivot['outlet'] == outlet)]
        if not row.empty:
            n = int(row['total'].values[0])
            ax.text(i, 102, f'n={n}', ha='center', va='bottom', fontsize=7, color='#555555')

axes[0].set_ylabel('% of sentences')
handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=10)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('fig1_bert_sentiment_by_year.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_bert_sentiment_by_year.png")

# Fig 2: Outlet profiles across all years
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Overall Outlet Profiles — Trump Coverage (All Years)", fontsize=13, fontweight='bold')

outlets_in_data = ['CNN', 'Fox News', 'NYT']
op = outlet_profile_pivot.set_index('outlet').reindex(outlets_in_data)
bottom = np.zeros(len(outlets_in_data))
for label in ['negative', 'neutral', 'positive']:
    col = f'{label}_pct'
    vals = op[col].fillna(0).values if col in op.columns else np.zeros(len(outlets_in_data))
    ax.bar(outlets_in_data, vals, bottom=bottom, label=label.capitalize(),
           color=COLORS[label], edgecolor='white', linewidth=0.5)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v >= 6:
            ax.text(i, b + v / 2, f'{v:.0f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
    bottom += vals

# Sample size above bar
for i, outlet in enumerate(outlets_in_data):
    row = outlet_profile_pivot[outlet_profile_pivot['outlet'] == outlet]
    if not row.empty:
        n = int(row['total'].values[0])
        ax.text(i, 102, f'n={n}', ha='center', va='bottom', fontsize=8, color='#555555')

ax.set_ylabel('% of sentences')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 112)
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('fig2_outlet_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_outlet_profiles.png")

# Fig 3: BERT net score vs GDELT AvgTone (line chart per outlet)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("BERT Net Score vs GDELT AvgTone by Outlet & Year", fontsize=13, fontweight='bold')

outlet_colors = {'AP': '#1f77b4', 'CNN': '#ff7f0e', 'Fox News': '#d62728', 'NYT': '#9467bd'}

# Ensure correct year order
year_order = ['2016', '2020', '2024']

for outlet in OUTLETS:
    sub = comparison[comparison['outlet'] == outlet].copy()
    sub['year'] = pd.Categorical(sub['year'], categories=year_order, ordered=True)
    sub = sub.sort_values('year')
    if sub.empty:
        continue
    axes[0].plot(sub['year'], sub['bert_net_score'], marker='o',
                 label=outlet, color=outlet_colors.get(outlet), linewidth=2)
    axes[1].plot(sub['year'], sub['avg_tone'], marker='o',
                 label=outlet, color=outlet_colors.get(outlet), linewidth=2)

    # Value labels on points
    for _, row in sub.iterrows():
        axes[0].annotate(f"{row['bert_net_score']:.1f}",
                         (row['year'], row['bert_net_score']),
                         textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=7, color=outlet_colors.get(outlet))
        axes[1].annotate(f"{row['avg_tone']:.2f}",
                         (row['year'], row['avg_tone']),
                         textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=7, color=outlet_colors.get(outlet))

axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0].set_title('BERT Net Score\n(% positive − % negative)')
axes[0].set_ylabel('Net score')
axes[0].set_xlabel('Year')
axes[0].set_xticks(year_order)

axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[1].set_title('GDELT AvgTone')
axes[1].set_ylabel('AvgTone')
axes[1].set_xlabel('Year')
axes[1].set_xticks(year_order)

for ax in axes:
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('fig3_bert_vs_gdelt.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_bert_vs_gdelt.png")

print("\nAll done!")
