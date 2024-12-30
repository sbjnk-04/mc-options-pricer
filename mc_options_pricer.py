import numpy as np

# The technique to actually price the options is to then calculate the exercise value ("payoff") of the option for each path, which will then be averaged
# and discounted to today.
def opt_pricer(stock, strike, time, rfr, sig, sim, time_steps=10, option_type="call"): #for European Pricing !!!
    dt = time/time_steps 
    sim_stock = np.zeros((time_steps, sim))
    sim_stock[0] = stock

    for t in range(1, time_steps):
        z = np.random.normal(0, 1, sim) 
        drift = (rfr - 0.5 *sig**2) * dt
        diffusion = sig * z * np.sqrt(dt)
        sim_stock[t] = sim_stock[t-1] * np.exp(drift + diffusion)
    
    if t == 1:
            print(f"Drift term (first path): {drift}")
            print(f"Diffusion term (first path): {diffusion[:10]}") 

    print(f"Simulated stock prices at maturity (first 10 paths): {sim_stock[-1][:10]}")

    if option_type.upper() == "CALL":
        payoff = np.maximum(sim_stock[-1]-strike, 0)
    elif option_type.upper() == "PUT":
        payoff = np.maximum(strike-sim_stock[-1], 0)
    else:
        raise ValueError("Invalid Option Type.")
    option_price = np.exp(-rfr * time)*np.mean(payoff)
    return option_price 

stock = 102.1
strike = 105.73
time = 0.9877432
rfr = 0.03
sig = 0.15
sim = 8000

call_price = opt_pricer(stock, strike, time, rfr, sig, sim,  option_type="CALL")
put_price = opt_pricer(stock, strike, time, rfr, sig, sim,  option_type="PUT")

print(f"Call Option Price = {round(call_price, 2)}")
print(f"Put Option Price = {round(put_price, 2)}")

