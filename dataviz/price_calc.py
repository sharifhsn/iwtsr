import scipy.stats as st

f0 = 0.035
k = 0.035
vol = 0.02
t = 0.5
annuity = 0.48

d = (f0 - k) / (vol * (t**0.5))
price = annuity * (vol * (t**0.5)) * (d * st.norm.cdf(d) + st.norm.pdf(d))
print(f"Theoretical Bachelier Price: {price}")
