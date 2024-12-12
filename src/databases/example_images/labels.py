from os import listdir
import shutil

families=["benjamin","berbew","ceeinject","dinwod","ganelp","gepys"]#,"mira","musecador","sfone","sillyp2p","small","upatre","wabot","wacatac"]

c=["client1","client2","client3","client4","client5","client6","client7","client8","server"]
client="client5/"
for client in c:
    with open(client+"/labels.csv","w+") as f:
        for ind, family in enumerate(families):
            files=listdir(client+"/"+family)        
            for file in filesa: 
                f.write(f"{family}/{file}, {ind}\n") #client+family


