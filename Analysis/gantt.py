import plotly.express as px
import pandas as pd

df = pd.DataFrame([
    dict(Task="Advanced topics in graphical deep learning", Start='2023-01-01', Finish='2023-04-30', Resource='Course work'),
    dict(Task="Workshop NeurIPS", Start='2023-05-01', Finish='2023-11-15', Resource='Conference'),
    dict(Task="Main conference CVPR", Start='2023-11-18', Finish='2024-03-28', Resource='Conference'),
    dict(Task="Main conference NeurIPS", Start='2024-05-19', Finish='2024-09-14', Resource='Conference'),
    dict(Task="Thesis writing", Start='2024-10-01', Finish='2025-07-01', Resource='Thesis'),
    dict(Task="Thesis defense", Start='2025-07-01', Finish='2025-08-01', Resource='Thesis'),
    dict(Task="Colour features", Start='2023-01-01', Finish='2023-06-01', Resource='Concept milestone'),
    dict(Task="Shape features", Start='2023-06-01', Finish='2023-12-31', Resource='Concept milestone'),
    dict(Task="Model work", Start='2024-01-01', Finish='2024-06-01',Resource='Concept milestone'),
    dict(Task="Generalization to other CV tasks", Start='2024-06-01', Finish='2025-04-01',Resource='Concept milestone'),
])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", title='Gantt chart of future plans', color='Resource')
fig.update_yaxes(autorange="reversed", tickfont_size=20) # otherwise tasks are listed from the bottom up
fig.show()