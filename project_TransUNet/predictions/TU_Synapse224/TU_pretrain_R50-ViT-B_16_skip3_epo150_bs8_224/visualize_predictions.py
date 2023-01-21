# %%
from medpy.io.load import load
import numpy as np 
from matplotlib import pyplot as plt 
import imageio

# %%
for case_number in ["01","02","03","04","08","25","38"]:
    data = load("D:\\TransUNet_Analysis\\project_TransUNet\\predictions\\TU_Synapse224\\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_224\\case00"+ case_number +"_img.nii.gz")
    label = load("D:\\TransUNet_Analysis\\project_TransUNet\\predictions\\TU_Synapse224\\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_224\\case00"+ case_number +"_gt.nii.gz")
    pred = load("D:\\TransUNet_Analysis\\project_TransUNet\\predictions\\TU_Synapse224\\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_224\\case00"+ case_number +"_pred.nii.gz")
    # %%
    image_data = data[0]
    image_label = label[0]
    image_pred = pred[0]

    print(image_data.shape,image_label.shape)
    d_label = {}
    d_label["1"] = "spleen"
    d_label["2"] = "right kidney"
    d_label["3"] = "left kidney"
    d_label["4"] = "gallbladder"
    d_label["5"] = "esophagus"
    d_label["6"] =  "liver"
    d_label["7"] = "stomach"
    d_label["8"] = "aorta"
    d_label["9"] = "inferior vena cava"
    d_label["10"] = "portal vein and splenic vein"
    d_label["11"] = "pancreas"
    d_label["12"] = "right adrenal gland"
    d_label["13"] = "left adrenal gland"

    # %%
    for n in range(image_data.shape[-1]):
        print("At case", case_number," at iteration ",n)
        plt.figure(figsize=(45,15))
        organs = np.unique(image_label[:,:,n])
        organs = np.setdiff1d(organs,[0])
        title_str = ", ".join( [d_label[str(int(i))].upper() for i in organs] )
        ax1 = plt.subplot(1,3,1)
        plt.imshow(image_data[:,:,n])
        plt.title("Image",fontsize=16)
        ax2 = plt.subplot(1,3,2)
        plt.suptitle("Slice: "+str(n)+"\nClasses: "+title_str,fontsize=25)
        plt.imshow(image_label[:,:,n])
        plt.title("Ground Truth",fontsize=16)
        ax2 = plt.subplot(1,3,3)
        plt.suptitle("Slice: "+str(n)+"\nClasses: "+title_str,fontsize=25)
        plt.imshow(image_pred[:,:,n])
        plt.title("Prediction",fontsize=16)
        # plt.show(block=False)
        plt.savefig(f'D:\\TransUNet_Analysis\\project_TransUNet\\predictions\\gif_predict\\slice_{n}.png', 
                    transparent = False  
                )
        # plt.pause(0.01)
        # ax1.clear()
        # ax2.clear()
        plt.close()
    
    # %%
    frames = []
    for t in range(image_data.shape[-1]):
        image = imageio.v2.imread(f'D:\\TransUNet_Analysis\\project_TransUNet\\predictions\\gif_predict\\slice_{t}.png')
        frames.append(image)
    # %%
    imageio.mimsave('D:\\TransUNet_Analysis\\project_TransUNet\\predictions\\gif_predict\\example'+case_number+'.gif', # output gif
                    frames,          # array of input frames
                    fps = image_data.shape[-1]//15)  
    # %%
