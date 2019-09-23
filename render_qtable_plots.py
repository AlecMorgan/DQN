"""
Having a model that works isn't good enough.
If we don't understand how our model is able
to work, then it's entirely possible that it
has found some way to "cheat" the objective--
an extremely common problem in reinforcement
learning, as it turns out. 

By visualizing some of the inner workings of
our model, we increase its interpretability. 
Interpretability helps us to fix problems, 
make improvements, and last but not least to
explain why our model chose to do one thing
instead of another. This last point is esp-
ecially important in contexts such as bus-
iness in which companies can be held liable
for the choices made by the AI systems that
they choose to deploy. 
"""

import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('ggplot')


def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

for i, file_name in enumerate(os.listdir("qtables"), start=1):
    q_table = np.load(f"qtables/{file_name}")

    if i % 100 == 0:
        # TODO(Alec): Complete plotting refactor using array masking instead of for loops. 
        # for n, ax in enumerate(axes, start=0):
        #     x, y = q_table[n]
        #     color, alpha = get_q_color(y_vals[n], y_vals)
        #     ax.scatter(x, y, c=color, marker="o", alpha=alpha)
        #     ax.set_ylabel(f"Action {n}")


        for x, x_vals in enumerate(q_table):
            for y, y_vals in enumerate(x_vals):
                for n, ax in enumerate(axes, start=0):
                    color, alpha = get_q_color(y_vals[n], y_vals)
                    ax.scatter(x, y, c=color, marker="o", alpha=alpha)
                    ax.set_ylabel(f"Action {n}")

        plt.savefig(f"qtable_charts/{i}")