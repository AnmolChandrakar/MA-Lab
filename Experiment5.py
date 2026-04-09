import pandas as pd

# Define the dataset
data = {
    'weather': ['Sunny', 'Sunny', 'Rainy', 'Rainy'],
    'playgame': ['Yes', 'Yes', 'No', 'No']
}
df = pd.DataFrame(data)

print("Dataset:")
display(df)

# Calculate Prior Probabilities
# P(playgame=Yes) and P(playgame=No)

prior_yes = len(df[df['playgame'] == 'Yes']) / len(df)
prior_no = len(df[df['playgame'] == 'No']) / len(df)

print(f"Prior Probability P(playgame=Yes): {prior_yes:.2f}")
print(f"Prior Probability P(playgame=No): {prior_no:.2f}")

# Calculate Likelihoods
# P(weather=Sunny | playgame=Yes)
# P(weather=Rainy | playgame=Yes)
# P(weather=Sunny | playgame=No)
# P(weather=Rainy | playgame=No)

# For playgame=Yes
df_yes = df[df['playgame'] == 'Yes']
likelihood_sunny_given_yes = len(df_yes[df_yes['weather'] == 'Sunny']) / len(df_yes)
likelihood_rainy_given_yes = len(df_yes[df_yes['weather'] == 'Rainy']) / len(df_yes)

# For playgame=No
df_no = df[df['playgame'] == 'No']
likelihood_sunny_given_no = len(df_no[df_no['weather'] == 'Sunny']) / len(df_no)
likelihood_rainy_given_no = len(df_no[df_no['weather'] == 'Rainy']) / len(df_no)

print(f"Likelihood P(weather=Sunny | playgame=Yes): {likelihood_sunny_given_yes:.2f}")
print(f"Likelihood P(weather=Rainy | playgame=Yes): {likelihood_rainy_given_yes:.2f}")
print(f"Likelihood P(weather=Sunny | playgame=No): {likelihood_sunny_given_no:.2f}")
print(f"Likelihood P(weather=Rainy | playgame=No): {likelihood_rainy_given_no:.2f}")

# Classify for input: weather = Sunny
input_weather = 'Sunny'

# Calculate Posterior Probabilities
# P(playgame=Yes | weather=Sunny) = P(weather=Sunny | playgame=Yes) * P(playgame=Yes) / P(weather=Sunny)
# P(playgame=No | weather=Sunny) = P(weather=Sunny | playgame=No) * P(playgame=No) / P(weather=Sunny)

# Denominator P(weather=Sunny) is the sum of numerators
# P(weather=Sunny) = P(weather=Sunny | playgame=Yes) * P(playgame=Yes) + P(weather=Sunny | playgame=No) * P(playgame=No)

prob_sunny = (likelihood_sunny_given_yes * prior_yes) + (likelihood_sunny_given_no * prior_no)

posterior_yes_given_sunny = (likelihood_sunny_given_yes * prior_yes) / prob_sunny
posterior_no_given_sunny = (likelihood_sunny_given_no * prior_no) / prob_sunny

print(f"\nFor input weather = {input_weather}:")
print(f"Posterior Probability P(playgame=Yes | weather=Sunny): {posterior_yes_given_sunny:.2f}")
print(f"Posterior Probability P(playgame=No | weather=Sunny): {posterior_no_given_sunny:.2f}")

# Classification
if posterior_yes_given_sunny > posterior_no_given_sunny:
    prediction = 'Yes'
else:
    prediction = 'No'

print(f"\nClassification: Given weather is {input_weather}, the playgame prediction is: {prediction}")
