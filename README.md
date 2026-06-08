# AI-ML
#1.This gave us our first glimpse into what the data looks like. We immediately saw that it tracks specific machine assets alongside operational metrics and business metrics.
#2.We learned the dataset is exactly 500 rows long and 13 columns wide. It contains 8 numerical columns (like cost and hours) and 5 categorical/text columns (like location and inventory level).
#3.We learned the boundaries of our data. 
#4.We found 0 missing values.
#5.There were 0 duplicates. Every single row represents a unique record or moment in time for the supply chain.
#6.This organized our approach for the visualizations.
#7.Both histograms showed a very flat, "uniform" distribution. Instead of a normal bell curve where most values sit in the middle, the temperatures and costs were spread very evenly across all possible ranges.
#8.Inventory levels are perfectly balanced. However, the Efficiency Label is imbalanced.
#9.the costs remained almost exactly the same regardless of whether inventory was Low, Medium, or High.
#10.The resulting graph looked like a random cloud of dots. This proved there is no direct linear relationship between maxing out resources and getting a highly efficient delivery.
#11.This was the biggest discovery of the EDA: almost every square on the heatmap was close to 0.00. This means none of the numerical features (like temperature, vibration, or downtime) have a simple linear relationship with one another.
#12.The shape of the "violins" were nearly identical for labels 0, 1, and 2. This showed us that the physical condition score of an asset doesn't dictate whether the supply chain will be labeled efficient or inefficient.
#13-14.This step taught us how to translate raw data and charts into actionable business intelligence—specifically concluding that the data is very clean, highly randomized, lacks linear relationships, and will require complex, non-linear machine learning models in the future.
