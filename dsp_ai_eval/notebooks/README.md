The notebooks in this folder capture bits of analysis that didn't get refactored into pipeline steps either because they were tangential or because time ran out!

`gpt_headings.ipynb`
This notebook does some simple frequency calculations as to how often different theme headings appear under different GPT models/temperatures.

`gpt_temps.ipynb`
This notebook visualises the embedded summaries with different GPT temperatures shown in different colours. You can modify how many different temperatures are visualised together. The aim was to see if the same themes cropped up under different temperature settings, or if, for example, summaries produced at temperature 0 covered a smaller area (in vector space) compared to those produced at higher temperatures.

`plot_together.ipynb`
This notebook concatenates the vector representations of the GPT-produced summaries, and the vector representations of the research abstracts, and plots them together.
