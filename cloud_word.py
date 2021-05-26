# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
#% matplotlib inline

# Load in the dataframe
df = pd.read_csv("archive/winemag-data-130k-v2.csv", index_col=0)
# Looking at first 5 rows of the dataset
print(df.head())

print("There are {} observations and {} features in this dataset. \n".format(df.shape[0],df.shape[1]))

print("There are {} types of wine in this dataset such as {}... \n".format(len(df.variety.unique()),
                                                                           ", ".join(df.variety.unique()[0:5])))

print("There are {} countries producing wine in this dataset such as {}... \n".format(len(df.country.unique()),
            ", ".join(df.country.unique()[0:5])))

print(df[["country", "description","points"]].head())

# Groupby by country
country = df.groupby("country")

# Summary statistic of all countries
print(country.describe().head())

print(country.mean().sort_values(by="points",ascending=False).head())

#tracer le nombre de vins par pays
# plt.figure(figsize=(15,10))
# country.size().sort_values(ascending=False).plot.bar()
# plt.xticks(rotation=50)
# plt.xlabel("Country of Origin")
# plt.ylabel("Number of Wines")
# #plt.show()

#un coup d'œil à l'intrigue des 44 pays par son vin le mieux noté
# plt.figure(figsize=(15,10))
# country.max('points').sort_values(by="points",ascending=False)["points"].plot.bar()
# plt.xticks(rotation=50)
# plt.xlabel("Country of Origin")
# plt.ylabel("Highest point of Wines")
# #plt.show()

# Start with one review:
#text = df.description[0]

# # Create and generate a word cloud image:
# wordcloud = WordCloud().generate(text)

# # Display the generated image:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()       

# lower max_font_size, change the maximum number of word and lighten the background:
# wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()
# # Save the image in the img folder:
# wordcloud.to_file("first_review.png")

text = " ".join(review for review in df.description)
print ("There are {} words in the combination of all review.".format(len(text)))

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# Generate a word cloud image
#wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# # Display the generated image:
# # the matplotlib way:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

# wine_mask = np.array(Image.open("B_V_.png"))
# print(wine_mask)
#Tout d'abord, vous utilisez la transform_format()fonction pour permuter le numéro 0 à 255.

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val

# # Transform your mask into a new one that will work with the function:
# transformed_wine_mask = np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)

# for i in range(len(wine_mask)):
#     transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))

# print(transformed_wine_mask)

# # Create a word cloud image
# wc = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask,
#                stopwords=stopwords, contour_width=3, contour_color='firebrick')

# # Generate a wordcloud
# wc.generate(text)

# # store to file
# wc.to_file("iB_V_.png")

# # show
# plt.figure(figsize=[20,10])
# plt.imshow(wc, interpolation='bilinear')
# plt.axis("off")
# plt.show()

country.size().sort_values(ascending=False).head()
country.size().sort_values(ascending=False).head(10)

# Join all reviews of each country:
usa = " ".join(review for review in df[df["country"]=="US"].description)
fra = " ".join(review for review in df[df["country"]=="France"].description)
ita = " ".join(review for review in df[df["country"]=="Italy"].description)
spa = " ".join(review for review in df[df["country"]=="Spain"].description)
por = " ".join(review for review in df[df["country"]=="Portugal"].description)

# Generate a word cloud image

mask = np.array(Image.open("fr3.jpeg"))
print(mask)
# replace 0 with 255
# mask[mask == 255] = 0
# print(mask)
#mask3 = np.where(mask2==0, 255, mask2)
#mask = np.where (mask == 0, 255, mask)

#wordcloud_usa = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA", max_words=100, mask=mask).generate(usa)
wordcloud_usa = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA", max_words=1000, mask=mask).generate(fra)

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[20,20])
plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# store to file
#plt.savefig("us_wine.png", format="png")
plt.savefig("fr_wine04.png", format="png")