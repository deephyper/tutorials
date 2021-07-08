import matplotlib.pyplot as plt
from dh_project.polynome2.model_run import run

point = {"lr": 0.01, "activation": "linear", "units": 10}
objective = run(point)
print("objective: ", objective)

from dh_project.polynome2.model_run import HISTORY

plt.plot(HISTORY["val_r2"])
plt.xlabel("Epochs")
plt.ylabel("Objective: $R^2$")
plt.grid()
plt.show()