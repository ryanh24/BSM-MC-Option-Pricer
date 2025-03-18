# Option-Pricer with Different Models
## by Ryan Har
This started off as a mini project to introduce myself to the idea of the Black Scholes Model Option pricing and Monte Carlo Option pricing, two concepts that I had heard of but didn't know much about/how to implement it. 
I decided that it would be useful for me to see how these different option pricers compare to the current market value of an option, in order to practice implementation and just build on my knowledge. My hope is to include other pricing models as well, more rigorous ones, and expand my knowledge.
## Instructions
### Sidebar: simply enter the ticker of the company you want to examine, click Enter
### Main: You can adjust the date and the range of strike values that you can look at, I am keeping the controls pretty simple. Then you can adjust the values of r and IV to change the outputs of the option pricers. They will show up in a DataFrame so you can compare the values, and by changing the sliders, you can observe the price discrepancies.

### Code: I chose to hide certain things in the DataFrame, which you can change in the rename_dict as well as under get_options_data(). I chose a MC with an antithetic variable as a control variate. This is definitely a continual work in progress.
