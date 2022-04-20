
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set(style="darkgrid", font_scale=2.5)
sns.set(style="darkgrid", font_scale=1.0)

gammas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.99]
total_cost = reversed([
    312.51418030859454,
    312.52389238304584,
    313.0185979073204,
    313.38734736529004,
    313.40386132684687,
    313.53024127687456])

fig, ax = plt.subplots(1)
random_total_cost = [322.12632582703907 for _ in range(len(gammas))]
naive_total_cost = [313.5794151617258 for _ in range(len(gammas))]


#sns.lineplot(ax=ax, x=gammas, y=total_cost, marker="o", linewidth=5, markersize=10)
sns.lineplot(ax=ax, x=gammas, y=total_cost, linewidth=3)
# sns.lineplot(ax=ax, x=gammas, y=random_total_cost, marker="o", linewidth=5, markersize=10)
sns.lineplot(ax=ax, x=gammas, y=naive_total_cost)


labels = ["Value Lookhead Strategy", "Naive"]
lgd = ax.legend(labels=labels)

ax.set_xlabel("Gamma", fontsize=12)
ax.set_ylabel("Total Cost", fontsize=12)

plt.savefig("figures/gamma_results.png", transparent=False,  dpi=300, bbox_inches='tight')
plt.show()