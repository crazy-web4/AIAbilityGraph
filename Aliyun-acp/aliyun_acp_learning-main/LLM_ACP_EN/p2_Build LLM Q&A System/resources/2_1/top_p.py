import numpy as np
from matplotlib import pyplot as plt

# Define a user input prompt and possible next words
prompt = "In the large model ACP course, you can learn"
next_words = ["RAG(token1)", "Prompt(token2)", "Model(token3)", "Writing(token4)", "Drawing(token5)"]
probabilities = [0.5, 0.24, 0.12, 0.09, 0.05]

# Calculate cumulative probabilities to illustrate top-p sampling
cumulative_probabilities = np.cumsum(probabilities)

# Define plot titles based on temperature scenario
titles = ["top_p=0.5", "top_p=0.9"]
top_p_thresholds = [0.5, 0.9]

plt.rcParams['font.family'] = 'Alibaba PuHuiTi'
fig, axs = plt.subplots(1, 2, figsize=(25, 6), sharey=True)

for i, ax in enumerate(axs):
    top_p_threshold = top_p_thresholds[i]

    # Identify which words fall under top-p threshold
    selected_indices = np.where(cumulative_probabilities <= top_p_threshold)[0]
    selected_words = np.array(next_words)[selected_indices]
    selected_probs = np.array(probabilities)[selected_indices]

    # Plot the general probability distribution first
    rects1 = ax.bar(next_words, probabilities, color='lightgrey', edgecolor='black', alpha=0.6, label='Unselected tokens')

    # Highlight the words chosen under the top-p threshold
    rects2 = ax.bar(selected_words, selected_probs, color='skyblue', edgecolor='black', alpha=0.9, label='Selected tokens')

    # Adding labels to the bars
    ax.bar_label(rects1, padding=5)
    ax.bar_label(rects2, padding=5)

    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel("Candidate tokens", fontsize=12, labelpad=13)  # Increase the padding for xlabel
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

fig.suptitle("The impact of parameter top_p on the sampling range of candidate tokens", fontsize=16)
plt.show()