# %%
import autoLabel as al
import pickle
import matplotlib.pyplot as plt

# Three days of example data of an espresso machine
with open("test.data",'rb') as f:
    dataDict = pickle.load(f)

# We can use either Active (p), Reactive (q) or Apparent (s) power to detect events
power = dataDict["data"]["p"]
sr = dataDict["samplingrate"]

threshold = 5.0 # Power jumps we are interested in (watt)
# All windows to use in seconds
preEventTime = 2.0 
postEventTime = 2.5
votingTime = 2.0
minDistance = 2.0 # min distance between events
m = 0.005 # Linear value for which threshold is increased as t_i = m*mean(preEventWindow_i) + threshold

events, labels = al.autoLabel(
            power, sr, 
            thres=5.0, preEventTime=preEventTime, postEventTime=postEventTime, votingTime=votingTime, 
            minDistance=minDistance, m=m, 
            verbose=True)

# Plot all events
plt.plot(power)
for e, label in zip(events, labels):
    plt.axvline(x=e, color=(0,0,0))

plt.show()

# %%
# Lets zoom in to an active section
zoomedPower = power[35000:36000]
events, labels = al.autoLabel(
            zoomedPower, sr, 
            thres=5.0, preEventTime=preEventTime, postEventTime=postEventTime, votingTime=votingTime, 
            minDistance=minDistance, m=m, 
            verbose=True)

fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
ax.plot(zoomedPower)
for e, label in zip(events, labels):
    ax.axvline(x=e, color=(0,0,0))
    y = ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])/1.3
    ax.text(e, y, label, rotation=270)

ax.set_ylabel("Power [W]")
ax.set_xlabel("Sample")
plt.show()


# %%
