from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'results/stage5b1_frontier_final.csv')
# Scenario-optimal baselines
fig=plt.figure(figsize=(7.2,4.8)); ax=fig.add_subplot(111)
for d,n,label in [('uniform','homogeneous','homogeneous / uniform H3'),('activity_specific','activity_scaled','activity-scaled / activity-specific H3')]:
 s=df[(df.design==d)&(df.noise==n)].sort_values('effective_power_authority_MW')
 ax.plot(s.effective_power_authority_MW/1000,s.Wplus,marker='o',label=label)
ax.set_xlabel('effective completion-authority proxy (GW)')
ax.set_ylabel(r'exact-PIC completed value $\mathcal{J}^{+}$')
ax.set_title('Case118 exact-PIC reinforcement frontier, scenario-matched baselines')
ax.legend(); ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(ROOT/'figures/stage5b1_scenario_optimal_frontier.pdf');fig.savefig(ROOT/'figures/stage5b1_scenario_optimal_frontier.png',dpi=180);plt.close(fig)
# Robust fixed comparison
fig=plt.figure(figsize=(7.2,4.8)); ax=fig.add_subplot(111)
for n,label in [('homogeneous','homogeneous'),('activity_scaled','activity-scaled')]:
 s=df[(df.design=='robust_fixed')&(df.noise==n)].sort_values('effective_power_authority_MW')
 ax.plot(s.effective_power_authority_MW/1000,s.Wplus,marker='o',label=label)
ax.set_xlabel('effective completion-authority proxy (GW)')
ax.set_ylabel(r'exact-PIC completed value $\mathcal{J}^{+}$')
ax.set_title('Case118 exact-PIC reinforcement frontier, robust-fixed metric')
ax.legend(); ax.grid(True,alpha=.25); fig.tight_layout()
fig.savefig(ROOT/'figures/stage5b1_robust_frontier.pdf');fig.savefig(ROOT/'figures/stage5b1_robust_frontier.png',dpi=180);plt.close(fig)
