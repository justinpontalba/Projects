# %%
import os
from PIL import Image
import pandas as pd


# %%
img_manifest = pd.read_csv(r"[set path here]\XRAY Dataset\Images\indiana_projections.csv")
img_manifest = img_manifest[img_manifest["projection"] == "Frontal"]
img_manifest.head(10)


# %%
dir_path = r"[set path here]\XRAY Dataset\Images\images\images_normalized\\"
for image in img_manifest["filename"]:
    img = Image.open(dir_path + image)
    img.save(r"[set path here]\XRAY Dataset\Images\frontal\\" + image)

# %%
df_reports = pd.read_csv(r"[set path here]\XRAY Dataset\Images\indiana_reports.csv")
df_reports.head(10)
# %%
frontal_dir = sorted(os.listdir(r"[set path here]\XRAY Dataset\Images\frontal\\"))
uid_list = []
report_list =[]
image_list = []
dataset = {}
for uid, findings, impression in zip(df_reports["uid"], df_reports["findings"], df_reports["impression"]):
    for image in frontal_dir:
        uid_frontal_dir = image.split("_")[0]
        if str(uid) == uid_frontal_dir:
            
            if str(findings) == "nan":
                findings = ""
            if str(impression) == "nan":
                impression = ""

            # dataset[image] ={"report": findings + impression}
            uid_list.append(uid)
            report_list.append("sos " + findings + impression + " eos"  )
            image_list.append("[set path here]/XRAY Dataset/Images/frontal/" + image)

df_dataset = pd.DataFrame({"uid": uid_list, "report": report_list, "image": image_list})
df_dataset.head(10)

# %%
from sklearn.model_selection import train_test_split

X = df_dataset["image"]
y = df_dataset["report"]

# Split the data - 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

test_df = pd.DataFrame({"image": X_test, "report": y_test})
train_df= pd.DataFrame({"image": X_train, "report": y_train})

test_df.to_csv(r"[set path here]\XRAY Dataset\Images\Data Prep\test.csv", index=False)
train_df.to_csv(r"[set path here]\XRAY Dataset\Images\Data Prep\train.csv", index=False)

# %%
train_dict ={}
for image, report in zip(train_df["image"], train_df["report"]):
    train_dict[image] = report

test_dict ={}
for image, report in zip(test_df["image"], test_df["report"]):
    test_dict[image] = report

# save the dictionary as json
import json
json_train = json.dumps(train_dict)
f = open(r"[set path here]\XRAY Dataset\Images\Data Prep\train.json","w")
f.write(json_train)

json_test = json.dumps(test_dict)
f = open(r"[set path here]\XRAY Dataset\Images\Data Prep\test.json","w")
f.write(json_test)

# %%
