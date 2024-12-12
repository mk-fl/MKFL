from os import listdir
import shutil

families=["benjamin","berbew","ceeinject","dinwod","ganelp","gepys","mira","musecador","sfone","sillyp2p","small","upatre","wabot","wacatac"]
for family in families:
    #files=listdir("bodmas1_images_couleur/"+family)
    files=listdir("bodmas2_Images_Couleur/"+family)
    for i in range(8):
        selection=files[i:i+46]#[i:i+41]
        for file in selection:
            shutil.copy("bodmas2_Images_Couleur/"+family+"/"+file,"client"+str(i+1)+"/"+family)

    selection=files[368:]#[328:500]
    for file in selection:
        shutil.copy("bodmas2_Images_Couleur/"+family+"/"+file,"server/"+family)
