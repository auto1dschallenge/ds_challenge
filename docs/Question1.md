# Question 1

## Task:
List as many use cases for the dataset as possible.

## Answer

The dataset contains 4 different kinds of information:
1. The price of a car (_price_)
2. The normalized losses per year of use (_normalized-losses_)
3. An actuarial risk rating relative to the normal risk assigend by price (_symboling_)
4. Data specifying the individual car (_all other_)

### Possible use cases

##### Predicting the prize of a car
Knowing the price a car will obtain once sold is obviously very important information for Auto1. 
Knowing the selling price of a car determines for how much the car can be bought given constrains on the margin/profit that one wants to obtain with the car.

#### Assesing the normalized-losses
The normalized-losses of a car can be interpreted as the value depreciation of a specific car over the course of one year in use.
Thus knowing the normalized-losses can be helpful to predict the future price of a car should it not be bought now, but a year from now.
Assuming that a seller knows about the normalized-losses of a car, this information can also be used to gain bargaining power. 
E.g. getting a lower price for a car right now, because the seller might incur additional costs if he is not able to sell right now.

#### Predicting the risk
I assume the risk rating of a car influences the insurance costs of a car. I.e. high risk leads to higher insurance fees.
As such the _symboling_ represents "hidden costs" for the owner of a car that are not represented in the price of a car. 
Thus the risk could also influence demand for a car, as higher _symboling_ could make a car undesirable for a potential buyer when compared to other cars with the same price.

#### Finding Under/Overvalued Cars (in relation to risk)
Both normalized-losses and the symboling represent information about the risk associated with a car. Assuming that high risk should correlate to larger normalized-losses, one could use the dataset to find outliers in the ratio. E.g. cars which incur high losses but who's actuarial risk is undervalued.
This could be used to find cars whose insurance costs are undeservedely low when compared to cars with similar normalized-losses.

#### Autocomplete missing specifications in individual car data
When a potential seller uploads data on a car to Auto1, this dataset could be used to automatically predict datapoints the seller did not fill in, therefore allowing to better asses the quality/price of a car.