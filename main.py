import numpy as np
import matplotlib.pyplot as plt

from kf2 import KF

file=open('kalmann.txt','r')
f=file.readline() #iterating over next line
f=file.readline() #iterating over next line
kf = KF(1,-0.0990697000865573, 6.361599147637872,0.1 ,368.18931566716645,6.5966106132139)

x=np.zeros(259)
x_original=np.zeros(259)
y_original=np.zeros(259)
y=np.zeros(259)
time=np.linspace(1,259,259)
t=0
for step in range(259):
    f=file.readline()
    list1=f.split(" ,")
    x_from_data=float(list1[0])
    x_original[t]=float(list1[0])
    y_from_data=float(list1[1])
    y_original[t]=float(list1[1])
    vx_from_data=float(list1[2])
    vy_from_data=float(list1[3])
    kf.predict()
    z=np.matrix([[x_from_data+vx_from_data],[y_from_data+vy_from_data]])
    kf.update(z)
    x[t]=kf.x[0][0]
    y[t]=kf.x[1][0]
    t=t+1

print("ending x position is :",kf.x[0][0])
print("ending y position is :",kf.x[1][0])
print("final uncertainity is :",kf.P[0])
figure,axis=plt.subplots(2,1)

axis[0].plot(x,y)

axis[0].set_title("x vs y")

axis[1].plot(x_original,y_original)

axis[1].set_title("x original vs y original")
plt.show()
