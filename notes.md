## Retreving Data 
For the data, I believe doing it individually for each stock would be the most benfital. We want as much data for each stock so forecasting will be accurate.
So lets loop through each ticker grabbing max data. One problem I see with the optimization is that some stocks simply have more data than others, but this is fine because in an optimal portfolio its unlikely we would want newer stocks. 

## Learning
I will test out my data pipline using a basic lstm model to forecast. 

from this website [time-series-forecasting, geeksforgeeks](https://www.geeksforgeeks.org/data-analysis/time-series-forecasting-using-pytorch/). It's not that amazing but it should show off the idea.

## Paper notes for model

"Only 70% of the dataset was allocated to the training set, while the remaining 15% was reserved for validation, and testing received the remaining 15%."

"The data were standardized using the MinMaxScaler function to increase the model's efficacy and prevent scaling discrepancies"

Added the business ratios, scaling, and the different data splits.


make it put the checkpoints for a specific tickers into the checkpoints folder